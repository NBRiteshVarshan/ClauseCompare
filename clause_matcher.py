import time
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ollama
from utils import generate_clause_id

class LegalClauseMatcher:
    """Match legal clauses using embeddings and LLM verification"""

    def __init__(self, embedding_model: str = 'nomic-ai/nomic-embed-text-v1.5'):
        """Initialize the matcher with embedding model"""
        print(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        self.cache = {}
        self.llm_model = 'qwen2.5:7b'

    def get_embeddings(self, clauses: List[Dict], doc_name: str) -> np.ndarray:
        """Get embeddings for clauses with caching"""
        embeddings = []
        for clause in clauses:
            clause_id = generate_clause_id(clause['text'], doc_name)
            if clause_id in self.cache:
                embeddings.append(self.cache[clause_id])
            else:
                embedding = self.embedder.encode(clause['text'])
                self.cache[clause_id] = embedding
                embeddings.append(embedding)
        return np.array(embeddings)

    def find_candidates(self, query_embedding: np.ndarray,
                       target_embeddings: np.ndarray,
                       k: int = 5) -> Tuple[List[int], List[float]]:
        """Find top-k candidate matches using cosine similarity"""
        similarities = cosine_similarity([query_embedding], target_embeddings)[0]
        top_indices = np.argsort(similarities)[-k:][::-1]
        return top_indices.tolist(), similarities[top_indices].tolist()

    def compare_with_llm(self, clause_a: str, clause_b: str) -> Dict[str, Any]:
        """Compare two clauses using LLM"""
        try:
            prompt = """You are comparing two legal clauses. They match if:
1. They create the same obligation (who must do what)
2. They grant the same right or power
3. They have the same conditions and exceptions
4. The consequences of violation are the same

They DO NOT match if:
- One has additional conditions the other lacks
- The scope is different (e.g., "worldwide" vs "US only")
- The obligation is stronger/weaker (e.g., "shall" vs "may")

Clause A: "{clause_a}"
Clause B: "{clause_b}"

Respond in JSON format with these keys:
- "match": true/false (boolean)
- "confidence": 0-1 (float)
- "key_differences": ["difference1", "difference2"] (list)
- "reason": "brief explanation" (string)"""

            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {'role': 'system', 'content': 'You are a legal text comparison expert. Respond only in JSON format.'},
                    {'role': 'user', 'content': prompt.format(clause_a=clause_a[:500], clause_b=clause_b[:500])}
                ]
            )
            result = json.loads(response['message']['content'])
            return result
        except Exception as e:
            print(f"LLM comparison error: {str(e)}")
            return {
                'match': False,
                'confidence': 0.5,
                'key_differences': ['LLM comparison failed'],
                'reason': 'Fallback to embedding similarity'
            }

    def match_documents(self, doc1_clauses: List[Dict], doc2_clauses: List[Dict],
                       doc1_name: str = "Document 1", doc2_name: str = "Document 2",
                       similarity_threshold: float = 0.3,
                       high_similarity_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Match clauses between two documents.

        Args:
            similarity_threshold: Minimum similarity to consider a candidate for LLM check.
            high_similarity_threshold: If similarity >= this value, treat as match without LLM.
        """
        start_time = time.time()

        print("Generating embeddings...")
        doc1_embeddings = self.get_embeddings(doc1_clauses, doc1_name)
        doc2_embeddings = self.get_embeddings(doc2_clauses, doc2_name)

        matched_doc2 = [False] * len(doc2_clauses)
        only_in_doc1 = []
        matching_details = []

        print("Comparing clauses...")
        total_clauses = len(doc1_clauses)

        for i, clause1 in enumerate(doc1_clauses):
            candidates, similarities = self.find_candidates(
                doc1_embeddings[i], doc2_embeddings, k=5
            )

            top_sim = similarities[0] if similarities else 0.0
            top_idx = candidates[0] if candidates else -1

            found_match = False
            best_match = None

            for j, (candidate_idx, similarity) in enumerate(zip(candidates, similarities)):
                if similarity < similarity_threshold:
                    continue

                # Fast‑path for high similarity
                if similarity >= high_similarity_threshold:
                    found_match = True
                    best_match = {
                        'clause_idx': candidate_idx,
                        'similarity': similarity,
                        'confidence': 1.0,
                        'reason': 'High similarity match (≥0.8)',
                        'key_differences': [],
                        'used_llm': False
                    }
                    matched_doc2[candidate_idx] = True
                    break

                # Use LLM for verification
                clause2 = doc2_clauses[candidate_idx]
                llm_result = self.compare_with_llm(clause1['text'], clause2['text'])

                if llm_result.get('match', False) and llm_result.get('confidence', 0) > 0.7:
                    found_match = True
                    best_match = {
                        'clause_idx': candidate_idx,
                        'similarity': similarity,
                        'confidence': llm_result.get('confidence', 0),
                        'reason': llm_result.get('reason', 'LLM match'),
                        'key_differences': llm_result.get('key_differences', []),
                        'used_llm': True
                    }
                    matched_doc2[candidate_idx] = True
                    break

            print(f"Progress: {i+1}/{total_clauses} (top sim: {top_sim:.3f})")

            matching_details.append({
                'clause_number': clause1.get('number', str(i+1)),
                'clause_text': clause1['text'],
                'found_match': found_match,
                'best_match': best_match,
                'top_similarity': top_sim,
                'top_match_idx': top_idx
            })

            if not found_match:
                # We have the closest from top_idx
                closest_idx = top_idx
                closest_sim = top_sim
                only_in_doc1.append({
                    'text': clause1['text'],
                    'number': clause1.get('number', str(i+1)),
                    'closest_match': doc2_clauses[closest_idx]['text'] if closest_idx >= 0 else "",
                    'similarity': closest_sim,
                    'metadata': clause1.get('metadata', {})
                })

        # Compute best match for doc2 clauses for categorization
        doc2_best_similarities = []
        for j, clause2 in enumerate(doc2_clauses):
            sims = cosine_similarity([doc2_embeddings[j]], doc1_embeddings)[0]
            best_idx = np.argmax(sims)
            best_sim = sims[best_idx]
            doc2_best_similarities.append(best_sim)

        # Find clauses only in doc2 (not matched)
        only_in_doc2 = []
        for j, (clause2, is_matched) in enumerate(zip(doc2_clauses, matched_doc2)):
            if not is_matched:
                # best match from doc1
                best_sim = doc2_best_similarities[j]
                # find index again for closest text
                sims = cosine_similarity([doc2_embeddings[j]], doc1_embeddings)[0]
                best_idx = np.argmax(sims)
                only_in_doc2.append({
                    'text': clause2['text'],
                    'number': clause2.get('number', str(j+1)),
                    'closest_match': doc1_clauses[best_idx]['text'] if best_idx >= 0 else "",
                    'similarity': best_sim,
                    'metadata': clause2.get('metadata', {})
                })

        processing_time = time.time() - start_time

        llm_matches = sum(1 for m in matching_details if m['found_match'] and m['best_match'] and m['best_match'].get('used_llm', True))
        high_sim_matches = sum(1 for m in matching_details if m['found_match'] and m['best_match'] and not m['best_match'].get('used_llm', True))

        print(f"✅ Matches: {len([m for m in matching_details if m['found_match']])} total "
              f"({high_sim_matches} via high-similarity, {llm_matches} via LLM)")

        return {
            'only_in_doc1': only_in_doc1,
            'only_in_doc2': only_in_doc2,
            'matching_details': matching_details,
            'total_doc1': len(doc1_clauses),
            'total_doc2': len(doc2_clauses),
            'matching_count': sum(1 for m in matching_details if m['found_match']),
            'processing_time': processing_time,
            'similarity_threshold': similarity_threshold,
            'high_similarity_threshold': high_similarity_threshold,
            'llm_matches': llm_matches,
            'high_sim_matches': high_sim_matches,
            'doc2_best_similarities': doc2_best_similarities
        }