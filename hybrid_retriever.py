# hybrid_retriever.py - 完整版本，包装新retriever并保持所有接口兼容

from pathlib import Path
import logging
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import get_paths

logger = logging.getLogger(__name__)

def normalize(scores):
    """Normalize scores to 0-1 range"""
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score - min_score == 0:
        return [1.0 for _ in scores]
    return [(s - min_score) / (max_score - min_score) for s in scores]

class Retriever:
    """
    包装UnifiedArxivRetriever以完全兼容旧接口
    """
    
    def __init__(self, e5_index_directory, bm25_index_directory, 
                 top_k=5, strategy="hybrid", alpha=0.65):
        
        logger.info(f"Initializing {strategy.upper()} retriever...")
        
        # 使用新的统一检索器
        try:
            from unified_arxiv_retriever import UnifiedArxivRetriever
            self._inner = UnifiedArxivRetriever(
                e5_index_directory=e5_index_directory,
                bm25_index_directory=bm25_index_directory,
                leveldb_path=None,  
                strategy=strategy,
                alpha=alpha,
                top_k=top_k
            )
            self._using_new = True
            logger.info("Using optimized UnifiedArxivRetriever")
        except Exception as e:
            logger.warning(f"Failed to load UnifiedArxivRetriever: {e}")
            self._using_new = False
            # 这里可以添加降级逻辑
            raise
        
        # 保持兼容的属性
        self.strategy = strategy
        self.alpha = alpha
        self.top_k = top_k
        
        # 初始化兼容性属性（即使不使用）
        self._doc_cache = {}
        self._abstract_cache = {}
        self._fulltext_cache = {}
        self._cache_size = 100
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._retrieval_times = []
        
        # 兼容旧代码的属性
        self.e5 = self._inner.e5 if hasattr(self._inner, 'e5') else None
        self.bm25 = self._inner.bm25 if hasattr(self._inner, 'bm25') else None
        self._bm25_retriever = self.bm25  # 别名
        
    def retrieve_abstracts(self, query: str, top_k: int = None) -> list:
        """检索摘要 - 保持接口不变"""
        if top_k is None:
            top_k = self.top_k
            
        start_time = time.time()
        
        # 检查缓存
        cache_key = f"{self.strategy}_{query.lower().strip()}_{top_k}"
        if cache_key in self._abstract_cache:
            logger.info(f"⚡ {self.strategy.upper()} cache hit!")
            return self._abstract_cache[cache_key]
        
        logger.info(f"🔍 {self.strategy.upper()} retrieval for query: {query}")
        
        # 使用新retriever
        result = self._inner.retrieve_abstracts(query, top_k)
        
        # 缓存结果
        if len(self._abstract_cache) >= self._cache_size:
            oldest_key = next(iter(self._abstract_cache))
            del self._abstract_cache[oldest_key]
        self._abstract_cache[cache_key] = result
        
        # 记录时间
        elapsed = time.time() - start_time
        self._retrieval_times.append(elapsed)
        if len(self._retrieval_times) > 100:
            self._retrieval_times = self._retrieval_times[-100:]
        
        logger.info(f"{self.strategy.upper()}: Retrieved {len(result)} abstracts in {elapsed:.2f}s")
        
        return result
    
    def get_full_texts(self, doc_ids: list, db=None) -> list:
        """获取全文 - 保持接口不变"""
        if not doc_ids:
            return []
        
        start_time = time.time()
        logger.info(f"{self.strategy.upper()}: Retrieving full texts for {len(doc_ids)} documents")
        
        # 使用新retriever
        result = self._inner.get_full_texts(doc_ids, db)
        
        elapsed = time.time() - start_time
        if result:
            total_chars = sum(len(text) for text, _ in result)
            avg_length = total_chars // len(result)
            logger.info(f"{self.strategy.upper()}: Retrieved {len(result)} full texts in {elapsed:.2f}s (avg {avg_length} chars/doc)")
        
        return result
    
    def retrieve(self, query: str, top_k: int = None):
        """旧版retrieve方法 - 为了兼容"""
        if top_k is None:
            top_k = self.top_k
            
        # 获取abstracts
        abstracts = self.retrieve_abstracts(query, top_k)
        
        # 转换为旧格式
        results = []
        for i, (text, doc_id) in enumerate(abstracts):
            results.append({
                "id": doc_id,
                "paper_id": doc_id,
                "title": "Unknown",  
                "abstract": text,
                "semantic_score": 1.0 / (i + 1)  # 简单的排名分数
            })
        
        return results
    
    def close(self):
        """清理资源"""
        try:
            if hasattr(self._inner, 'close'):
                self._inner.close()
        except:
            pass
        
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
        
        self._doc_cache.clear()
        self._abstract_cache.clear()
        self._fulltext_cache.clear()
        
        stats = self.get_performance_stats()
        if "avg_retrieval_time" in stats:
            logger.info(f"Final stats: {stats['total_retrievals']} retrievals, avg {stats['avg_retrieval_time']:.2f}s")
        
        logger.info("Retriever closed")
    
    def get_bm25_status(self):
        """诊断方法"""
        if self._using_new:
            return {"method": "optimized", "available": True, "status": "FAST"}
        else:
            return {"method": "unknown", "available": False, "status": "UNKNOWN"}
    
    def get_performance_stats(self):
        """性能统计"""
        if self._retrieval_times:
            avg_time = sum(self._retrieval_times) / len(self._retrieval_times)
            return {
                "avg_retrieval_time": avg_time,
                "total_retrievals": len(self._retrieval_times),
                "cache_sizes": {
                    "abstract_cache": len(self._abstract_cache),
                    "fulltext_cache": len(self._fulltext_cache),
                    "doc_cache": len(self._doc_cache),
                },
                "bm25_status": self.get_bm25_status()
            }
        return {"no_data": True}
    
    # 以下是一些可能被调用的内部方法
    def _fast_normalize(self, scores):
        """快速归一化"""
        if len(scores) == 0:
            return scores
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score - min_score == 0:
            return np.ones_like(scores)
        return (scores - min_score) / (max_score - min_score)
    
    def _get_bm25_results(self, query, top_k):
        """兼容方法 - 如果有代码直接调用这个"""
        if self.bm25:
            # 调用新retriever的BM25
            return self._inner._retrieve_bm25(query, top_k)
        return []
    
    def _get_e5_results(self, query, top_k):
        """兼容方法 - 如果有代码直接调用这个"""  
        if self.e5:
            # 调用新retriever的E5
            docs = self._inner.e5.retrieve(query, top_k)
            # 转换格式
            return docs
        return []
    
    def _load_bm25_into_memory(self):
        """兼容方法 - 返回None因为新系统自动处理"""
        return None