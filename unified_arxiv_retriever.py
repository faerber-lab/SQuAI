#!/usr/bin/env python3
"""
统一检索器 - E5 + BM25混合检索
基于测试结果优化，适配你的arXiv索引结构
"""

import json
import sqlite3
import logging
import time
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer
import torch  # Add this import

logger = logging.getLogger(__name__)


class E5DirectRetriever:
    """
    优化版: 使用缓存映射，避免每次重新构建
    第一次运行需要5分钟构建映射，之后只需要几秒钟加载
    """
    
    def __init__(self, index_directory: str, model_name: str = "intfloat/e5-large-v2"):
        self.index_dir = Path(index_directory)
        self.model_name = model_name
        
        # 映射缓存文件路径
        self.mapping_cache_file = self.index_dir / "faiss_document_mapping.pkl"
        
        logger.info(f"初始化E5检索器: {index_directory}")
        
        # 加载FAISS索引
        self._load_index()
        
        # 加载或构建映射
        if self.mapping_cache_file.exists():
            self._load_cached_mapping()
        else:
            logger.info("首次运行，需要构建映射（约5分钟）...")
            self._connect_db()
            self._build_and_save_mapping()
        
        # 加载并优化E5模型
        self._load_model()
        
        # 查询缓存
        self._cache = {}
        self._cache_size = 100
        
        logger.info(f"E5检索器初始化完成")
    
    def _load_index(self):
        """加载FAISS索引"""
        index_path = self.index_dir / "faiss_index"
        self.index = faiss.read_index(str(index_path))
        logger.info(f"FAISS索引加载: {self.index.ntotal:,} 向量, 维度 {self.index.d}")
    
    def _load_cached_mapping(self):
        """从缓存文件加载映射（快速）"""
        logger.info(f"从缓存加载映射: {self.mapping_cache_file}")
        start_time = time.time()
        
        try:
            with open(self.mapping_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Validate the loaded data
            if 'faiss_to_doc' not in cache_data or 'doc_metadata' not in cache_data:
                raise ValueError("Cache file is missing required data")
            
            self.faiss_to_doc = cache_data['faiss_to_doc']
            self.doc_metadata = cache_data['doc_metadata']
            
            elapsed = time.time() - start_time
            logger.info(f"✓ 映射加载完成: {len(self.faiss_to_doc):,} 文档 (耗时 {elapsed:.1f}秒)")
            
        except (EOFError, pickle.UnpicklingError, ValueError) as e:
            logger.warning(f"缓存文件损坏或不完整: {e}")
            logger.info("删除损坏的缓存，重新构建映射...")
            
            # Delete corrupted cache
            if self.mapping_cache_file.exists():
                self.mapping_cache_file.unlink()
            
            # Rebuild
            self._connect_db()
            self._build_and_save_mapping()
    
    def _connect_db(self):
        """连接SQLite数据库（仅在需要构建映射时）"""
        db_path = self.index_dir / "index_store.db"
        self.db_conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db_cursor = self.db_conn.cursor()
        
        # 验证数据库
        self.db_cursor.execute("SELECT COUNT(*) FROM document")
        doc_count = self.db_cursor.fetchone()[0]
        logger.info(f"SQLite数据库连接: {doc_count:,} 文档")
    
    def _build_and_save_mapping(self):
        """构建映射并保存到缓存文件（只需运行一次）"""
        logger.info("构建文档映射（这需要约5分钟，但只需要做一次）...")
        start_time = time.time()
        
        # 创建映射: FAISS索引 -> 文档信息
        self.faiss_to_doc = {}
        
        # 批量获取所有文档的vector_id映射（更快）
        logger.info("  1/3 获取vector_id映射...")
        self.db_cursor.execute("""
            SELECT id, vector_id, content 
            FROM document 
            WHERE vector_id IS NOT NULL
        """)
        
        # 使用fetchall一次获取所有数据（比逐行获取快）
        all_docs = self.db_cursor.fetchall()
        total_docs = len(all_docs)
        
        logger.info(f"  2/3 处理 {total_docs:,} 文档...")
        for i, (doc_id, vector_id_str, content) in enumerate(all_docs):
            faiss_idx = int(vector_id_str)
            self.faiss_to_doc[faiss_idx] = {
                'id': doc_id,
                'content': content
            }
            
            if (i + 1) % 100000 == 0:
                logger.info(f"    已处理 {i+1:,}/{total_docs:,} 文档...")
        
        # 加载元数据
        logger.info("  3/3 加载文档元数据...")
        self.doc_metadata = {}
        
        self.db_cursor.execute("""
            SELECT document_id, name, value 
            FROM meta_document
            WHERE name = 'paper_id'
        """)
        
        for doc_id, name, value in self.db_cursor.fetchall():
            if doc_id not in self.doc_metadata:
                self.doc_metadata[doc_id] = {}
            
            if value:
                value = value.strip('"')
            self.doc_metadata[doc_id]['paper_id'] = value
        
        # 保存到缓存文件
        logger.info(f"保存映射到缓存文件: {self.mapping_cache_file}")
        cache_data = {
            'faiss_to_doc': self.faiss_to_doc,
            'doc_metadata': self.doc_metadata,
            'created_at': time.time(),
            'doc_count': len(self.faiss_to_doc)
        }
        
        with open(self.mapping_cache_file, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # 关闭数据库连接（不再需要）
        self.db_conn.close()
        
        elapsed = time.time() - start_time
        logger.info(f"✓ 映射构建并保存完成 (耗时 {elapsed:.1f}秒)")
        logger.info(f"  下次运行将直接加载缓存，只需几秒钟！")
    
    def _load_model(self):
        """加载并优化E5模型"""
        logger.info(f"加载E5模型: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        # 尝试使用GPU加速
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info("✓ E5模型在GPU上运行 (快速)")
            self.device = 'cuda'
        else:
            self.model = self.model.cpu()
            logger.info("⚠ E5模型在CPU上运行 (较慢)")
            self.device = 'cpu'
        
        self.model.eval()
        
        # 预热模型
        with torch.no_grad():
            _ = self.model.encode("warmup", convert_to_numpy=True, show_progress_bar=False)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        使用正确的vector_id映射检索文档
        """
        # 检查缓存
        cache_key = f"{query}_{top_k}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        start_time = time.time()
        
        # 编码查询（E5需要"query: "前缀）
        query_text = f"query: {query}"
        
        with torch.no_grad():
            query_embedding = self.model.encode(
                query_text, 
                convert_to_numpy=True, 
                show_progress_bar=False,
                normalize_embeddings=True
            )
        
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # FAISS搜索
        distances, indices = self.index.search(query_embedding, top_k)
        
        # 使用正确的映射获取文档
        results = []
        for dist, faiss_idx in zip(distances[0], indices[0]):
            if faiss_idx == -1:
                continue
            
            # 使用vector_id映射
            if faiss_idx in self.faiss_to_doc:
                doc_info = self.faiss_to_doc[faiss_idx]
                doc_id = doc_info['id']
                content = doc_info['content']
                
                # 获取元数据
                metadata = self.doc_metadata.get(doc_id, {})
                paper_id = metadata.get('paper_id', doc_id)
                
                results.append({
                    "id": doc_id,
                    "content": content,
                    "score": float(dist),
                    "paper_id": paper_id,
                    "metadata": metadata
                })
            else:
                logger.debug(f"FAISS索引 {faiss_idx} 未找到映射")
        
        # 缓存结果
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = results
        
        elapsed = time.time() - start_time
        
        if elapsed > 5:
            logger.warning(f"E5检索较慢: {elapsed:.3f}秒")
        else:
            logger.info(f"E5检索: {len(results)} 结果, 耗时 {elapsed:.3f}秒")
        
        return results
    
    def retrieve_abstracts(self, query: str, top_k: int = 5) -> List[Tuple[str, str]]:
        """
        检索摘要（兼容接口）
        
        Returns:
            List of (abstract_text, paper_id) tuples
        """
        docs = self.retrieve(query, top_k)
        return [(doc["content"], doc["paper_id"]) for doc in docs]
    
    def rebuild_mapping_cache(self):
        """强制重建映射缓存（如果索引更新了）"""
        logger.info("强制重建映射缓存...")
        
        # 删除旧缓存
        if self.mapping_cache_file.exists():
            self.mapping_cache_file.unlink()
            logger.info("已删除旧缓存文件")
        
        # 重新构建
        self._connect_db()
        self._build_and_save_mapping()
    
    def close(self):
        """清理资源"""
        if hasattr(self, 'db_conn'):
            self.db_conn.close()
        self._cache.clear()
        logger.info("E5检索器关闭")

class UnifiedArxivRetriever:
    """
    统一的arXiv检索器 - 支持E5、BM25和混合模式
    """
    
    def __init__(self, 
                 e5_index_directory: str,
                 bm25_index_directory: str, 
                 leveldb_path: str = None,
                 strategy: str = "hybrid",
                 alpha: float = 0.65,
                 top_k: int = 5):
        """
        初始化统一检索器
        
        Args:
            e5_index_directory: E5 FAISS索引目录
            bm25_index_directory: BM25 LlamaIndex索引目录
            leveldb_path: LevelDB全文存储路径（可选）
            strategy: "e5", "bm25", 或 "hybrid"
            alpha: E5权重 (hybrid模式)
            top_k: 默认返回文档数
        """
        self.strategy = strategy
        self.alpha = alpha
        self.top_k = top_k
        
        logger.info(f"初始化统一检索器 - 策略: {strategy}")
        
        # 初始化E5（如需要）
        self.e5 = None
        if strategy in ["e5", "hybrid"]:
            self.e5 = E5DirectRetriever(e5_index_directory)
            logger.info("E5检索器已加载")
        
        # 初始化BM25（如需要）
        self.bm25 = None
        if strategy in ["bm25", "hybrid"]:
            try:
                # 尝试使用你的fast实现
                from fast_llamaindex_retriever import FastLlamaIndexBM25Retriever
                self.bm25 = FastLlamaIndexBM25Retriever(bm25_index_directory, top_k)
                logger.info("BM25检索器已加载 (fast版本)")
            except ImportError:
                # 或使用你的BM25OnlyRetriever
                try:
                    from bm25_only_retriever import BM25OnlyRetriever
                    self.bm25 = BM25OnlyRetriever(bm25_index_directory, top_k)
                    logger.info("BM25检索器已加载 (纯BM25版本)")
                except ImportError:
                    logger.warning("BM25检索器不可用")
        
        # LevelDB（用于全文）
        self.leveldb = None
        if leveldb_path:
            try:
                import plyvel
                self.leveldb = plyvel.DB(leveldb_path, create_if_missing=False)
                logger.info(f"LevelDB已连接: {leveldb_path}")
            except Exception as e:
                logger.warning(f"LevelDB连接失败: {e}")
        
        # 并行执行器
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        # 缓存
        self._cache = {}
        self._cache_size = 50
    
    def retrieve_abstracts(self, query: str, top_k: int = None) -> List[Tuple[str, str]]:
        """
        检索摘要
        
        Returns:
            List of (abstract_text, paper_id) tuples
        """
        if top_k is None:
            top_k = self.top_k
        
        # 检查缓存
        cache_key = f"{self.strategy}_{query}_{top_k}"
        if cache_key in self._cache:
            logger.info("缓存命中")
            return self._cache[cache_key]
        
        logger.info(f"{self.strategy.upper()}检索: '{query}'")
        
        # 根据策略选择检索方法
        if self.strategy == "e5":
            result = self._retrieve_e5(query, top_k)
        elif self.strategy == "bm25":
            result = self._retrieve_bm25(query, top_k)
        else:  # hybrid
            result = self._retrieve_hybrid(query, top_k)
        
        # 缓存
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = result
        
        return result
    
    def _retrieve_e5(self, query: str, top_k: int) -> List[Tuple[str, str]]:
        """E5检索"""
        if not self.e5:
            logger.error("E5未初始化")
            return []
        return self.e5.retrieve_abstracts(query, top_k)
    
    def _retrieve_bm25(self, query: str, top_k: int) -> List[Tuple[str, str]]:
        """BM25检索"""
        if not self.bm25:
            logger.error("BM25未初始化")
            return []
        return self.bm25.retrieve_abstracts(query, top_k)
    
    def _retrieve_hybrid(self, query: str, top_k: int) -> List[Tuple[str, str]]:
        """混合检索"""
        if not self.e5 or not self.bm25:
            logger.error("混合模式需要E5和BM25")
            return self._retrieve_e5(query, top_k) if self.e5 else []
        
        # 并行检索
        e5_future = self._executor.submit(self.e5.retrieve, query, top_k * 2)
        bm25_future = self._executor.submit(self.bm25.retrieve_abstracts, query, top_k * 2)
        
        # 获取结果
        e5_results = e5_future.result()
        bm25_results = bm25_future.result()
        
        # 合并分数
        doc_scores = {}
        doc_contents = {}
        
        # E5结果
        for doc in e5_results:
            paper_id = doc["paper_id"]
            doc_scores[paper_id] = doc_scores.get(paper_id, 0) + self.alpha * doc["score"]
            doc_contents[paper_id] = doc["content"]
        
        # BM25结果
        for i, (content, paper_id) in enumerate(bm25_results):
            # BM25分数：使用排名倒数作为分数
            bm25_score = 1.0 / (i + 1)
            doc_scores[paper_id] = doc_scores.get(paper_id, 0) + (1 - self.alpha) * bm25_score
            if paper_id not in doc_contents:
                doc_contents[paper_id] = content
        
        # 排序并返回top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for paper_id, _ in sorted_docs[:top_k]:
            if paper_id in doc_contents:
                results.append((doc_contents[paper_id], paper_id))
        
        logger.info(f"混合检索: E5({len(e5_results)}) + BM25({len(bm25_results)}) = {len(results)} 结果")
        
        return results
    
    def get_full_texts(self, doc_ids: List[str], db=None) -> List[Tuple[str, str]]:
        """
        获取全文 - 修复：添加db参数以兼容RAG系统
        
        Args:
            doc_ids: 文档ID列表
            db: LevelDB实例（如果已经在外部打开）
        
        Returns:
            List of (full_text, doc_id) tuples
        """
        if not self.leveldb and db:
            # 如果没有内部leveldb但提供了外部db，使用外部的
            results = []
            for doc_id in doc_ids:
                try:
                    content = db.get(doc_id.encode('utf-8'))
                    if content:
                        results.append((content.decode('utf-8'), doc_id))
                except Exception as e:
                    logger.error(f"获取 {doc_id} 失败: {e}")
            return results
        elif self.leveldb:
            # 使用内部的leveldb
            results = []
            for doc_id in doc_ids:
                try:
                    content = self.leveldb.get(doc_id.encode('utf-8'))
                    if content:
                        results.append((content.decode('utf-8'), doc_id))
                except Exception as e:
                    logger.error(f"获取 {doc_id} 失败: {e}")
            return results
        else:
            logger.warning("没有可用的LevelDB")
            return []
    
    def close(self):
        """清理资源"""
        if self.e5:
            self.e5.close()
        if self.bm25 and hasattr(self.bm25, 'close'):
            self.bm25.close()
        if self.leveldb:
            self.leveldb.close()
        self._executor.shutdown(wait=True)
        logger.info("统一检索器已关闭")


# 使用示例
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 配置路径（根据你的实际路径调整）
    E5_INDEX = "/data/horse/ws/s3811141-faiss/inbe405h-unarxive/faiss_index"
    BM25_INDEX = "/data/horse/ws/s3811141-faiss/inbe405h-unarxive/bm25_retriever"
    LEVELDB = "/data/horse/ws/inbe405h-unarxive/full_text_db"  # 如果有的话
    
    # 测试不同策略
    strategies = ["e5", "bm25", "hybrid"] if len(sys.argv) <= 1 else [sys.argv[1]]
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"测试 {strategy.upper()} 策略")
        print('='*60)
        
        retriever = UnifiedArxivRetriever(
            e5_index_directory=E5_INDEX,
            bm25_index_directory=BM25_INDEX,
            leveldb_path=LEVELDB if Path(LEVELDB).exists() else None,
            strategy=strategy,
            alpha=0.65
        )
        
        # 测试查询
        queries = [
            "quantum computing algorithms",
            "deep learning transformers",
            "protein folding prediction"
        ]
        
        for query in queries:
            print(f"\n查询: '{query}'")
            start = time.time()
            results = retriever.retrieve_abstracts(query, top_k=3)
            elapsed = time.time() - start
            
            print(f"找到 {len(results)} 个结果 (耗时: {elapsed:.3f}秒)")
            for i, (abstract, paper_id) in enumerate(results, 1):
                preview = abstract[:100] + "..." if len(abstract) > 100 else abstract
                print(f"  [{i}] {paper_id}: {preview}")
        
        # 测试全文获取（如果配置了LevelDB）
        if retriever.leveldb and results:
            print("\n测试全文获取...")
            doc_ids = [paper_id for _, paper_id in results[:2]]
            full_texts = retriever.get_full_texts(doc_ids)
            for full_text, doc_id in full_texts:
                print(f"  {doc_id}: {len(full_text)} 字符")
        
        retriever.close()
    
    print("\n所有测试完成！")