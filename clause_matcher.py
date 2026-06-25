import time
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ollama
from utils import generate_clause_id

class LegalClauseMatcher:
    def __init__(self, embedding_model: str = 'nomic-ai/nomic-embed-text-v1.5'):
        print(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        self.cache = {}
        self.llm_model = 'qwen2.5:7b'

    def get_embeddings(self, clauses: List[Dict], doc_name: str) -> np.ndarray:
        embeddings = []
        for clause in clauses:
            cid = generate_clause_id(clause['text'], doc_name)
            if cid in self.cache:
                embeddings.append(self.cache[cid])
            else:
                emb = self.embedder.encode(clause['text'])
                self.cache[cid] = emb
                embeddings.append(emb)
        return np.array(embeddings)

    def find_candidates(self, query_emb, target_embs, k=5):
        sims = cosine_similarity([query_emb], target_embs)[0]
        top = np.argsort(sims)[-k:][::-1]
        return top.tolist(), sims[top].tolist()

    def compare_with_llm(self, clause_a, clause_b):
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
            return json.loads(response['message']['content'])
        except Exception as e:
            print(f"LLM error: {e}")
            return {'match': False, 'confidence': 0.5, 'key_differences': [], 'reason': 'LLM failed'}

    def match_documents(self, doc1_clauses, doc2_clauses,
                        doc1_name="Document 1", doc2_name="Document 2",
                        similarity_threshold=0.3, high_similarity_threshold=0.8):
        start = time.time()
        print("Generating embeddings...")
        emb1 = self.get_embeddings(doc1_clauses, doc1_name)
        emb2 = self.get_embeddings(doc2_clauses, doc2_name)

        matched_doc2 = [False] * len(doc2_clauses)
        only_in_doc1 = []
        matching_details = []

        print("Comparing clauses...")
        for i, clause1 in enumerate(doc1_clauses):
            candidates, sims = self.find_candidates(emb1[i], emb2, k=5)
            top_sim = sims[0] if sims else 0.0
            top_idx = candidates[0] if candidates else -1

            found = False
            best = None
            for j, (idx, sim) in enumerate(zip(candidates, sims)):
                if sim < similarity_threshold:
                    continue
                if sim >= high_similarity_threshold:
                    found = True
                    best = {'clause_idx': idx, 'similarity': sim, 'confidence': 1.0,
                            'reason': 'High similarity match (≥0.8)', 'key_differences': [], 'used_llm': False}
                    matched_doc2[idx] = True
                    break
                # LLM
                clause2 = doc2_clauses[idx]
                llm_res = self.compare_with_llm(clause1['text'], clause2['text'])
                if llm_res.get('match', False) and llm_res.get('confidence', 0) > 0.7:
                    found = True
                    best = {'clause_idx': idx, 'similarity': sim,
                            'confidence': llm_res.get('confidence', 0),
                            'reason': llm_res.get('reason', 'LLM match'),
                            'key_differences': llm_res.get('key_differences', []),
                            'used_llm': True}
                    matched_doc2[idx] = True
                    break

            print(f"Progress: {i+1}/{len(doc1_clauses)} (top sim: {top_sim:.3f})")
            matching_details.append({
                'clause_number': clause1.get('number', str(i+1)),
                'clause_text': clause1['text'],
                'found_match': found,
                'best_match': best,
                'top_similarity': top_sim,
                'top_match_idx': top_idx
            })
            if not found:
                closest = top_idx if top_idx >= 0 else 0
                only_in_doc1.append({
                    'text': clause1['text'],
                    'number': clause1.get('number', str(i+1)),
                    'closest_match': doc2_clauses[closest]['text'] if closest < len(doc2_clauses) else "",
                    'similarity': top_sim,
                    'metadata': clause1.get('metadata', {})
                })

        # Doc2 best similarities
        doc2_best_sims = []
        for j in range(len(doc2_clauses)):
            sims = cosine_similarity([emb2[j]], emb1)[0]
            doc2_best_sims.append(sims.max())

        only_in_doc2 = []
        for j, (clause2, matched) in enumerate(zip(doc2_clauses, matched_doc2)):
            if not matched:
                sims = cosine_similarity([emb2[j]], emb1)[0]
                best_idx = np.argmax(sims)
                only_in_doc2.append({
                    'text': clause2['text'],
                    'number': clause2.get('number', str(j+1)),
                    'closest_match': doc1_clauses[best_idx]['text'] if best_idx < len(doc1_clauses) else "",
                    'similarity': sims[best_idx],
                    'metadata': clause2.get('metadata', {})
                })

        elapsed = time.time() - start
        llm_count = sum(1 for m in matching_details if m['found_match'] and m['best_match'] and m['best_match'].get('used_llm', True))
        high_count = sum(1 for m in matching_details if m['found_match'] and m['best_match'] and not m['best_match'].get('used_llm', True))
        print(f"✅ Matches: {sum(1 for m in matching_details if m['found_match'])} total ({high_count} via high-sim, {llm_count} via LLM)")

        return {
            'only_in_doc1': only_in_doc1,
            'only_in_doc2': only_in_doc2,
            'matching_details': matching_details,
            'total_doc1': len(doc1_clauses),
            'total_doc2': len(doc2_clauses),
            'matching_count': sum(1 for m in matching_details if m['found_match']),
            'processing_time': elapsed,
            'similarity_threshold': similarity_threshold,
            'high_similarity_threshold': high_similarity_threshold,
            'llm_matches': llm_count,
            'high_sim_matches': high_count,
            'doc2_best_similarities': doc2_best_sims
        }