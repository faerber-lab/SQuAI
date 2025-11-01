#!/usr/bin/env python3
"""
hpc_remote_runner.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Automated helper that

1.  Parses command‑line arguments (`argparse`).
2.  Executes a command on an HPC head‑node via SSH, with automatic fail‑over to
    a fallback node after a configurable number of failures.
3.  Verifies the presence of Slurm tools (`sbatch`, `srun`, `squeue` …) on the
    remote host and checks whether the current user can log in password‑less.
4.  Keeps a local HPC script directory in sync with the remote target via
    `rsync --update --archive --delete`.
5.  Ensures that a Slurm job named *--hpc-job-name* is running for
    the current user; if not, submits `{local_hpc_script_dir}/slurm.sbatch`.
6.  Once the job is running, reads `--server_and_port_file` on the remote
    side, extracts *internal_host:port*, and sets up an SSH port‑forward so
    that a **local** HTTP endpoint transparently proxies to the remote server,
    optionally hopping through a jumphost.

The code attempts to stay robust and *loud* in its diagnostics, so you always
see what is happening.
"""

from __future__ import annotations

import argparse
import signal
import atexit
import asyncio
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path, PosixPath
from typing import Final, Optional
import getpass
import sys
from pprint import pprint
import psutil
from typing import Union
import getpass

from beartype import beartype
from rich.console import Console
from rich.table import Table
from rich.text import Text
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_fixed

host = None
port = None

try:
    from subprocess import Popen, DEVNULL, PIPE
except ImportError as err:  # pragma: no cover
    sys.exit(
        "Missing runtime dependencies.  Run `pip install -r requirements.txt` "
        f"first!  ({err})"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Globals & helpers
# ──────────────────────────────────────────────────────────────────────────────

console: Final = Console(highlight=False)

@beartype
def dier (msg: Any) -> None:
    pprint(msg)
    sys.exit(10)

@beartype
def rule (msg: str) -> None:
    if args.debug:
        console.rule(msg)
    else:
        console.print(f"→ {msg}")

@dataclass(slots=True)
class SSHConfig:
    target: str
    jumphost_url: str | None = None
    retries: int = 3
    debug: bool = False
    username: str | None = None
    jumphost_username: str | None = None

@beartype
def run_local(cmd: str, debug: bool = False, timeout: Optional[int] = 60) -> subprocess.CompletedProcess:
    """Run a *local* shell command, streaming output if `debug`."""
    if debug:
        console.log(f"[yellow]$ {cmd}")
    return subprocess.run(
        shlex.split(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )


@beartype
def build_ssh_cmd(
    cfg: SSHConfig,
    remote_cmd: str,
    allocate_tty: bool = False,
) -> str:
    """
    Compose an `ssh` command string with optional jumphost and debug flags.
    Uses ControlMaster auto‑socket for multiplexed connections (faster).
    """
    options = [
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=10",
        "-o", "ControlMaster=auto",
        "-o", "ControlPersist=60",
        #"-o", "ControlPath=~/.ssh/ctl-%r@%h:%p",
    ]
    if cfg.jumphost_url:
        options.extend(["-J", cfg.jumphost_url])

    if allocate_tty:
        options.append("-tt")  # force TTY allocation (Slurm sbatch often needs it)

    cmd = ["ssh", *options, cfg.target, remote_cmd]
    #print(" ".join(cmd))
    return " ".join(shlex.quote(a) for a in cmd)


@beartype
async def ssh_run(
    cfg: SSHConfig,
    remote_cmd: str,
    tty: bool = False,
) -> subprocess.CompletedProcess:
    """Async wrapper that retries a remote ssh command per cfg.retries."""
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(cfg.retries),
        retry=retry_if_exception_type(subprocess.CalledProcessError),
        wait=wait_fixed(3),
        reraise=True,
    ):
        with attempt:
            cp = run_local(build_ssh_cmd(cfg, remote_cmd, tty), debug=cfg.debug)
            if cp.returncode != 0:
                msg = f"SSH command failed (attempt {attempt.retry_state.attempt_number}): \n"
                msg += f"{remote_cmd}\n"
                msg += f"\n{cp.stderr.strip()}\n"
                console.print(f"[red]❌{msg}[/red]")
                raise subprocess.CalledProcessError(cp.returncode, cp.args, cp.stdout, cp.stderr)
            return cp
    raise RuntimeError("Unreachable")


# ──────────────────────────────────────────────────────────────────────────────
# core – step‑wise helper functions
# ──────────────────────────────────────────────────────────────────────────────

@beartype
async def verify_slurm_and_key(cfg: SSHConfig) -> None:
    """Check for Slurm commands & password‑less SSH."""
    rule("[bold]Verifying remote environment[/bold]")

    # test key auth
    cp = await ssh_run(cfg, "echo OK")
    if cp.stdout.strip() != "OK":
        console.print("[red]❌Password‑less SSH seems not configured.  Aborting.[/red]")
        sys.exit(1)
    console.print("[green]✓ Password‑less SSH works.[/green]")

    # check Slurm binaries
    slurm_tools = ["sbatch", "squeue", "srun"]
    missing: list[str] = []
    for tool in slurm_tools:
        cmd = f"command -v {shlex.quote(tool)} >/dev/null"
        cp = await ssh_run(cfg, cmd)
        if cp.returncode != 0:
            missing.append(tool)

    if missing:
        console.print(f"[red]❌Missing Slurm tools on {cfg.target}: {', '.join(missing)}[/red]")
        sys.exit(1)
    console.print("[green]✓ Slurm utilities present.[/green]")


@beartype
async def rsync_scripts(
    cfg: SSHConfig,
    local_dir: PosixPath,
    remote_dir: str | PosixPath,
) -> None:
    """Rsync local script directory to remote."""
    if not local_dir.is_dir():
        console.print(f"[red]❌ {local_dir} is not a directory.[/red]")
        sys.exit(1)

    user = getpass.getuser()

    console.rule(f"[bold]Ensuring remote directory exists[/bold] (running as user: {user})")

    mkdir_cmd = f"mkdir -p {shlex.quote(str(remote_dir))}"
    try:
        await ssh_run(cfg, mkdir_cmd)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ Failed to create remote directory:[/red] {e.stderr}")
        sys.exit(e.returncode)

    rule("[bold]Synchronising script directory[/bold]")

    target_user = args.username or getpass.getuser()

    # Prepare SSH command with optional jumphost
    ssh_cmd = "ssh"
    if args.jumphost_url:
        jumphost_user = args.jumphost_username or args.username or getpass.getuser()
        ssh_cmd += f" -J {jumphost_user}@{args.jumphost_url}"

    # Determine if cfg.target already includes username
    if "@" in cfg.target:
        remote_target = cfg.target
    else:
        remote_target = f"{target_user}@{cfg.target}"

    rsync_cmd = (
        f"rsync -az --delete "
        f"-e {shlex.quote(ssh_cmd)} "
        f"{shlex.quote(str(local_dir))}/ "
        f"{remote_target}:{shlex.quote(str(remote_dir))}/"
    )

    cp = run_local(rsync_cmd, debug=cfg.debug)
    if cp.returncode:
        console.print(f"[red]❌ rsync failed:[/red] {cp.stderr}")
        sys.exit(cp.returncode)

    console.print(f"[green]✓ {local_dir} → {remote_target}:{remote_dir} updated.[/green]")

@beartype
def to_absolute(path: str | PosixPath)  -> Path:
    """Convert a relative or absolute path to an absolute path."""
    return Path(path).expanduser().resolve()

@beartype
async def job_status_in_squeue(cfg: "SSHConfig") -> bool | None:
    """
    Prüft, ob der Job mit hpc_job_name in der squeue läuft.

    Rückgabe:
    - True  wenn mindestens ein Jobstatus RUNNING ist
    - False wenn kein Job RUNNING ist, aber mindestens einer PENDING ist
    - None  wenn kein Job RUNNING oder PENDING ist oder Job nicht in der squeue vorhanden ist
    """
    try:
        # Args werden hier angenommen global oder als Teil von cfg; am besten übergeben!
        job_name = shlex.quote(cfg.hpc_job_name) if hasattr(cfg, "hpc_job_name") else shlex.quote(args.hpc_job_name)

        list_cmd = f"squeue --me -h -o '%j|%T' | grep -F {job_name} || true"
        cp = await ssh_run(cfg, list_cmd)
        lines = cp.stdout.strip().splitlines()

        if len(lines) == 0:
            if hasattr(cfg, "debug") and cfg.debug:
                print("job_status_in_squeue: job not in squeue")
            return None

        found_pending = False
        for line in lines:
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 2:
                continue
            name, state = parts
            if name == (cfg.hpc_job_name if hasattr(cfg, "hpc_job_name") else args.hpc_job_name):
                if state == "RUNNING":
                    return True
                elif state == "PENDING":
                    found_pending = True
                else:
                    if hasattr(cfg, "debug") and cfg.debug:
                        print(f"job_status_in_squeue: Other status found: {state}")

        if found_pending:
            return False

        return None

    except Exception as e:
        print(f"❌Error in job_status_in_squeue for '{cfg.hpc_job_name if hasattr(cfg, 'hpc_job_name') else args.hpc_job_name}': {e}")
        return None

@beartype
async def ensure_job_running(
    cfg: "SSHConfig",
    remote_script_dir: PosixPath,
    heartbeat_msg: str = "Job already running"
) -> Union[bool, None]:
    """
    Ensures that the HPC job is running on the server.

    Returns:
    - True if job was started successfully,
    - False if job failed to start,
    - None if job was already running (no action needed).
    """
    try:
        if args.debug:
            rule("[bold]Ensuring server job is active[/bold]")

        job_name = args.hpc_job_name

        if await job_status_in_squeue(cfg) is not None:
            if args.debug:
                console.print(f"[green]✓ {heartbeat_msg}.[/green]")
            return None

        console.print("[yellow]Job not running – submitting…[/yellow]")

        sbatch_path = f"{remote_script_dir}/{args.sbatch_file_name}"
        submit_cmd = f"sbatch {shlex.quote(sbatch_path)}"
        cp = await ssh_run(cfg, submit_cmd, tty=True)

        job_id_line = cp.stdout.strip()
        console.print(job_id_line)

        try:
            job_id = int(job_id_line.strip().split()[-1])
        except Exception as e:
            console.print(f"[red]❌Failed to extract job ID: {e}[/red]")
            return False

        console.print("[cyan]Waiting for job to appear in queue or start…[/cyan]")

        timeout_seconds = 300
        poll_interval = 10
        elapsed = 0

        while elapsed < timeout_seconds:
            check_cmd = f"squeue -j {job_id} -h -o '%T'"
            cp = await ssh_run(cfg, check_cmd)
            job_state = cp.stdout.strip()

            if job_state == "RUNNING":
                console.print(f"[green]✓ Job is active: {job_state}[/green]")
                return True

            check_sacct = (
                f"sacct -j {job_id} --format=JobID,State --parsable2 --noheader || true"
            )
            cp = await ssh_run(cfg, check_sacct)

            for line in cp.stdout.strip().splitlines():
                parts = line.strip().split("|")
                if len(parts) >= 2 and parts[0].startswith(str(job_id)):
                    state = parts[1]
                    if state == "RUNNING":
                        console.print(f"[green]✓ Job is active in sacct: {state}[/green]")
                        return True
                    elif state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"):
                        console.print(f"[red]❌Job terminated early: {state}[/red]")
                        return False

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        console.print("[red]❌Timed out waiting for job to start or appear in system.[/red]")
        return False

    except Exception as e:
        console.print(f"[red]❌Unexpected error in ensure_job_running: {e}[/red]")
        return False


@beartype
async def wait_for_job_running_or_absent(cfg: "SSHConfig") -> bool | None:
    """
    Wartet, bis der Jobstatus True (RUNNING) ist oder None (nicht mehr in squeue).

    Prüft alle 5 Sekunden den Status.
    Gibt True zurück wenn RUNNING,
    None wenn Job nicht mehr in squeue ist.
    """
    poll_interval = 5

    while True:
        status = await job_status_in_squeue(cfg)
        if status is True:
            return True
        if status is None:
            return None
        await asyncio.sleep(poll_interval)

@beartype
async def read_remote_host_port(cfg: SSHConfig, primary_cfg: SSHConfig, fallback_cfg: Optional[SSHConfig]) -> Optional[tuple[str, int]]:
    """
    Poll remote server_and_port_file until it exists and contains "host:port",
    then parse and return it.
    """

    ret = await wait_for_job_running_or_absent(cfg)

    if ret is None:
        console.print(f"[red]❌The job seems to have been deleted.[/red]")
        await ensure_job_running(cfg, to_absolute(args.hpc_script_dir))
        await connect_and_tunnel(primary_cfg, fallback_cfg, args.local_hpc_script_dir)

    remote_path = args.server_and_port_file
    max_attempts = args.max_attempts_get_server_and_port
    delay_seconds = args.delay_between_server_and_port

    last_error: Optional[Exception] = None

    attempt = 1

    while attempt <= max_attempts:
        status = await job_status_in_squeue(cfg)

        if status is None:
            return None

        if status:
            try:
                cp = await ssh_run(cfg, f"cat {remote_path}")
                host_port = cp.stdout.strip()

                if not host_port:
                    raise RuntimeError("Empty response from remote file")

                host, port_s = host_port.split(":", 1)
                port = int(port_s)

                if args.debug:
                    console.print(f"[green]Remote server: {host}:{port}[/green]")

                return host, port

            except Exception as exc:
                last_error = exc
                console.print(f"[yellow]Waiting for remote host file ({attempt}/{max_attempts})…[/yellow]")
                await asyncio.sleep(delay_seconds)
                attempt += 1
        else:
            if args.debug:
                console.log(f"Job not yet running")
            await asyncio.sleep(delay_seconds)

    console.print(f"[red]❌Remote host file not found after {max_attempts} attempts[/red]")
    raise RuntimeError(
        f"Failed to read valid host:port from {remote_path} on {cfg.target} "
        f"after {max_attempts} tries"
    ) from last_error

@beartype
def find_process_using_port(port: int) -> Optional[tuple[int, str]]:
    try:
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr and conn.laddr.port == port:
                if conn.pid is not None:
                    try:
                        proc = psutil.Process(conn.pid)
                        return (proc.pid, proc.name())
                    except psutil.NoSuchProcess:
                        return (conn.pid, "Unknown")
        return None
    except Exception as e:
        return None

class SSHForwardProcess:
    def __init__(self, process: Popen, local_port: int, remote_host: str, remote_port: int):
        self.process = process
        self.local_port = local_port
        self.remote_host = remote_host
        self.remote_port = remote_port
        self._stopped = False

        # Registrieren für automatische Beendigung
        atexit.register(self.stop)

        # Ctrl+C abfangen, um auch dort zu stoppen
        signal.signal(signal.SIGINT, self._sigint_handler)
        signal.signal(signal.SIGTERM, self._sigterm_handler)

    def _sigint_handler(self, signum, frame):
        self.stop()
        # Jetzt wirklich Programm abbrechen
        os._exit(130)

    def _sigterm_handler(self, signum, frame):
        self.stop()
        os._exit(143)

    def stop(self):
        if self._stopped:
            return

        self._stopped = True

        if self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except Exception as e:
                console = Console()
                console.log(f"[red]❌Error while exiting Port-Forwardings: {e}[/red]")
        else:
            console = Console()
            console.log("[yellow]SSH-Forwarding-process ended already.[/yellow]")

@beartype
def start_port_forward(cfg, remote_host: str, remote_port: int, local_port: int) -> SSHForwardProcess:
    from rich.console import Console
    console = Console()

    if args.debug:
        rule("[bold]Starting Port Forwarding[/bold]")

    try:
        ssh_cmd_parts = [
            "ssh",
            "-L", f"{local_port}:{remote_host}:{remote_port}",
            "-N",
            "-T",
        ]

        # Falls ein Jumphost gesetzt ist, nutzen wir ihn über ProxyJump oder ProxyCommand
        if hasattr(cfg, "proxyjump") and cfg.proxyjump:
            ssh_cmd_parts += ["-J", cfg.proxyjump]
        #else:
        #    ssh_cmd_parts += ["-o", f"ProxyCommand=ssh -W %h:%p {shlex.quote(cfg.jumphost_url)}"]

        if hasattr(cfg, "identity_file") and cfg.identity_file:
            ssh_cmd_parts += ["-i", cfg.identity_file]

        ssh_cmd_parts.append(cfg.target)

        ssh_cmd_str = " ".join(shlex.quote(part) for part in ssh_cmd_parts)
        if args.debug:
            console.log(f"SSH-Forward-Command: {ssh_cmd_str}")

        process = Popen(
            ssh_cmd_parts,
            stdout=DEVNULL,
            stderr=PIPE,
            preexec_fn=os.setsid  # eigene Prozessgruppe für sauberes Beenden
        )

        # Kurzes Warten, um zu prüfen, ob Prozess korrekt startet
        import time
        time.sleep(1.0)
        if process.poll() is not None:
            err_output = process.stderr.read().decode("utf-8", errors="replace")
            console.print(f"[red]SSH-Forwarding failed:\n{err_output}. Command: {ssh_cmd_parts}[/red]")

        console.log(f"[green]Port forwarding is running: http://localhost:{local_port} -> {remote_host}:{remote_port}[/green]")
        return SSHForwardProcess(process, local_port, remote_host, remote_port)

    except Exception as e:
        console.log(f"[red]❌Error while trying to port-forward: {e}[/red]")
        raise

# ──────────────────────────────────────────────────────────────────────────────
# main entry‑point
# ──────────────────────────────────────────────────────────────────────────────

@beartype
def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Run or proxy the *--hpc-job-name* job on an HPC cluster.

            Examples
            --------
              # Basic usage
              python hpc_remote_runner.py \\
                  --hpc-system-url user@login.hpc.example.com \\
                  --fallback-system-url user@login2.hpc.example.com \\
                  --hpc-script-dir ./my_slurm_scripts \\
                  --copy --debug

              # Limit to one retry
              python hpc_remote_runner.py ... --retries 1
            """
        ),
    )

    parser.add_argument("--hpc-system-url", required=True, help="SSH target for primary HPC head-node (user@host)")
    parser.add_argument("--fallback-system-url", help="SSH target for fallback HPC head-node")
    parser.add_argument("--jumphost-url", help="Optional SSH jumphost in user@host form")
    parser.add_argument("--local-hpc-script-dir", required=True, type=Path, help="Local directory containing Slurm scripts")
    parser.add_argument("--hpc-script-dir", required=True, type=Path, help="Directory on the HPC System where the files should be copied to")
    parser.add_argument("--copy", action="store_true", help="If set, rsync the script directory before anything else")
    parser.add_argument("--debug", action="store_true", help="Verbose local shell output")
    parser.add_argument("--retries", type=int, default=3, help="SSH retry attempts before using fallback")
    parser.add_argument("--local-port", type=int, default=8000, help="Local port to expose the remote service")
    parser.add_argument("--heartbeat-time", type=int, default=10, help="Time to re-check if the server is still running properly")
    parser.add_argument("--username", default=getpass.getuser(), help="SSH username for HPC and (by default) also for jumphost")
    parser.add_argument("--sbatch_file_name", default="slurm.sbatch", help="Name of the file that contains the slurm job")
    parser.add_argument("--jumphost-username", help="SSH username for jumphost (defaults to --username)")
    parser.add_argument("--hpc-job-name", help="Name of the HPC job (defaults to slurm_runner)", default="slurm_runner")
    parser.add_argument("--server-and-port-file", help="Globally available path to a file where the hostname and port for the host should be put on HPC (defaults to ~/hpc_server_host_and_file)", default="~/hpc_server_host_and_file")
    parser.add_argument("--max-attempts-get-server-and-port", type=int, default=60, help="Number of attempts to get the --server-and-port-file from the HPC (defaults to 60)")
    parser.add_argument("--delay_between_server_and_port", type=int, default=5, help="Delay between calls to the --server-and-port-file check on HPC (defaults to 5)")

    parser.add_argument('--daemonize', action='store_true')

    return parser

@beartype
def kill_process(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
        if args.debug:
            console.log(f"Process {pid} was terminated with SIGKILL.")
    except ProcessLookupError:
        console.print(f"[red]❌ENo process with PID {pid} was found.[/red]")
    except PermissionError:
        console.print(f"[red]❌EInsufficient permissions to terminate process {pid}.[/red]")
    except Exception as e:
        console.print(f"[red]❌EUnexpected error while terminating process {pid}: {e}[/red]")

@beartype
async def run_with_host(cfg: SSHConfig, local_script_dir: Path, primary_cfg: SSHConfig, fallback_cfg: Optional[SSHConfig]) -> tuple[bool, Optional[SSHForwardProcess]]:
    global host, port

    """
    Execute the entire workflow for *one* remote host.

    Returns:
        (success: bool, fwd: SSHForwardProcess | None)
    """
    try:
        await verify_slurm_and_key(cfg)

        if args.copy and not os.path.exists("/etc/dont_copy"):
            await rsync_scripts(cfg, local_script_dir, args.hpc_script_dir)

        await ensure_job_running(cfg, to_absolute(args.hpc_script_dir))

        ret = await read_remote_host_port(cfg, primary_cfg, fallback_cfg)

        if ret is not None:
            host, port = ret

            fwd = start_port_forward(cfg, host, port, args.local_port)

            async def monitor_job():
                global host, port
                try:
                    while True:
                        await asyncio.sleep(args.heartbeat_time)
                        ret = await read_remote_host_port(cfg, primary_cfg, fallback_cfg)
                        if ret is not None:
                            new_host, new_port = ret

                            if await ensure_job_running(cfg, to_absolute(args.hpc_script_dir), "Heartbeat sent successfully") or new_host != host or new_port != port:
                                host = new_host
                                port = new_port

                                existing_proc_info = find_process_using_port(args.local_port)
                                if existing_proc_info:
                                    pid, name = existing_proc_info

                                    if args.debug:
                                        console.log(f"Local-Port is already used by process {pid} ({name}). Will kill it to restart it...")

                                    kill_process(pid)

                                    fwd = start_port_forward(cfg, host, port, args.local_port)
                        else:
                            console.print(f"[red]❌Remote job on was not in squeue anymore (B)[/red]")
                            ok, fwd = await run_with_host(primary_cfg, args.local_hpc_script_dir, primary_cfg, fallback_cfg)

                            return ok, fwd
                except Exception as e:
                    console.print(f"[red]❌Remote job on {cfg.target} appears to have stopped: {e}[/red]")
                    try:
                        fwd.stop()  # Beende Portweiterleitung, wenn Job weg
                    except Exception as e2:
                        console.print(f"[yellow]⚠️ Failed to stop forwarder cleanly: {e2}[/yellow]")

            # Starte Überwachungs-Task im Hintergrund
            asyncio.create_task(monitor_job())

            return True, fwd
        else:
            console.print(f"[red]❌Remote job on was not in squeue anymore (A)[/red]")

            return False, None

    except Exception as exc:  # noqa: BLE001
        console.print_exception()
        console.print(f"[red]❌Host {cfg.target} failed: {exc}[/red]")
        return False, None

async def main() -> None:
    global args
    parser = build_cli()
    args = parser.parse_args()

    if not args.jumphost_username:
        args.jumphost_username = args.username

    if args.daemonize:
        sys.stdout = open(os.devnull, 'w')

    rule(f"Checking if port is already in use")

    existing_proc_info = find_process_using_port(args.local_port)
    if existing_proc_info:
        pid, name = existing_proc_info
        console.print(f"[red]❌Local port {args.local_port} already used by PID {pid} ({name})[/red]")
        sys.exit(2)

    console.print(f"Starting with [bold]{args.hpc_system_url}[/bold]  (retries={args.retries})")

    target_url = f"{args.username}@{args.hpc_system_url}"
    jumphost_url = f"{args.jumphost_username}@{args.jumphost_url}" if args.jumphost_url else None

    primary_cfg = SSHConfig(
        target=target_url,
        jumphost_url=jumphost_url,
        retries=args.retries,
        debug=args.debug,
        username=args.username,
        jumphost_username=args.jumphost_username,
    )

    fallback_cfg = None
    if args.fallback_system_url:
        fallback_cfg = SSHConfig(
            target=f"{args.username}@{args.fallback_system_url}",
            jumphost_url=jumphost_url,
            retries=args.retries,
            debug=args.debug,
            username=args.username,
            jumphost_username=args.jumphost_username,
        )

    await connect_and_tunnel(primary_cfg, fallback_cfg, args.local_hpc_script_dir)

async def connect_and_tunnel(
    primary_cfg: SSHConfig,
    fallback_cfg: Optional[SSHConfig],
    local_hpc_script_dir: str
) -> None:
    # Versuch mit Haupt-Host
    ok, fwd = await run_with_host(primary_cfg, local_hpc_script_dir, primary_cfg, fallback_cfg)
    if ok:
        console.print("[bold green]✓  All done – tunnel is up.  Press Ctrl+C to stop.[/bold green]")
        try:
            while True:
                await asyncio.sleep(10)
        except KeyboardInterrupt:
            console.print("\n[cyan]Stopping tunnel…[/cyan]")
            fwd.stop()
            return

    # Falls Haupt-Host fehlschlägt und Fallback definiert
    if fallback_cfg is not None:
        console.print("[yellow]Trying fallback host…[/yellow]")
        ok, fwd = await run_with_host(fallback_cfg, local_hpc_script_dir, primary_cfg, fallback_cfg)
        if ok:
            console.print("[bold green]✓  All done – tunnel is up (fallback).  Press Ctrl+C to stop.[/bold green]")
            try:
                while True:
                    await asyncio.sleep(10)
            except KeyboardInterrupt:
                console.print("\n[cyan]Stopping tunnel…[/cyan]")
                fwd.stop()
                return
        else:
            console.print("[bold red]❌Both hosts failed.  Giving up.[/bold red]")
            sys.exit(1)

    else:
        console.print("[red]❌No fallback host defined. Use --fallback-system-url to define a fallback-host[/red]")
        if fwd is not None:
            fwd.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[red]❌You pressed CTRL-c or sent a signal. Program will end.[/red]")
