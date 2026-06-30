import time
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ollama
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import generate_clause_id

class LegalClauseMatcher:
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        self.cache = {}
        self.llm_model = 'llama3.2:3b'

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

    def compare_with_llm(self, clause_a, clause_b):
        """Synchronous LLM call – used by ThreadPoolExecutor."""
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
            return {'match': False, 'confidence': 0.0, 'reason': str(e)}

    def match_documents(self, doc1_clauses, doc2_clauses,
                        doc1_name="Document 1", doc2_name="Document 2",
                        similarity_threshold=0.4,
                        high_similarity_threshold=0.85,
                        match_threshold=0.5):

        start = time.time()
        print("Generating embeddings...")
        emb1 = self.get_embeddings(doc1_clauses, doc1_name)
        emb2 = self.get_embeddings(doc2_clauses, doc2_name)

        n1 = len(doc1_clauses)
        n2 = len(doc2_clauses)

        # Full similarity matrix
        sim_matrix = cosine_similarity(emb1, emb2)

        # Track matches – boolean array only (O(1) lookup)
        matched_doc2 = [False] * n2
        match_pairs = []  # list of (doc1_idx, doc2_idx, similarity, reason)

        # STEP 1: Mutual best matching
        print("Step 1: Finding mutual best matches...")
        best1 = np.argmax(sim_matrix, axis=1)
        best_sim1 = sim_matrix[np.arange(n1), best1]
        best2 = np.argmax(sim_matrix, axis=0)

        mutual_count = 0
        for i in range(n1):
            j = best1[i]
            sim = best_sim1[i]
            if best2[j] == i and sim >= match_threshold:
                if not matched_doc2[j]:
                    matched_doc2[j] = True
                    match_pairs.append((i, j, sim, 'Mutual best match'))
                    mutual_count += 1

        print(f"  → {mutual_count} mutual best matches found")

        # STEP 2: Greedy matching with high-similarity instant matches
        print("Step 2: Greedy matching remaining clauses...")
        instant_count = 0
        ambiguous_pairs = []  # list of (doc1_idx, doc2_idx, similarity)

        for i in range(n1):
            # Check if already matched via mutual best
            already_matched = False
            for pair in match_pairs:
                if pair[0] == i:
                    already_matched = True
                    break
            if already_matched:
                continue

            # Get top candidates for this doc1 clause
            sims = sim_matrix[i, :]
            sorted_indices = np.argsort(sims)[::-1]

            matched = False

            for j in sorted_indices:
                if matched_doc2[j]:
                    continue

                sim = sims[j]

                if sim < similarity_threshold:
                    break  # no more candidates above threshold

                if sim >= high_similarity_threshold:
                    # High similarity – instant match
                    matched_doc2[j] = True
                    match_pairs.append((i, j, sim, f'High similarity (≥{high_similarity_threshold})'))
                    matched = True
                    instant_count += 1
                    break

                elif sim >= similarity_threshold:
                    # Borderline – collect for LLM verification
                    ambiguous_pairs.append((i, j, sim))
                    # Don't break – we want to see all candidates
                    # but we only collect the best one per doc1 clause
                    # Actually, for proper matching we should collect all candidates
                    # but we'll process them in order of similarity

            # If we didn't find an instant match, we'll handle via LLM later
            # The ambiguous_pairs list contains all borderline candidates

        print(f"  → {instant_count} instant matches, {len(ambiguous_pairs)} ambiguous candidates")

        # STEP 3: Process ambiguous pairs with LLM (concurrent)
        if ambiguous_pairs:
            print(f"⚡ Processing {len(ambiguous_pairs)} ambiguous candidates with LLM...")

            # Sort by similarity descending (higher chance of match first)
            ambiguous_pairs.sort(key=lambda x: x[2], reverse=True)

            # Use ThreadPoolExecutor for concurrent LLM calls
            llm_matches = 0

            # First, filter ambiguous_pairs to only include candidates where doc2 is still unmatched
            # and doc1 is not yet matched
            filtered_pairs = []
            for i, j, sim in ambiguous_pairs:
                # Check if doc1 already matched
                doc1_matched = False
                for pair in match_pairs:
                    if pair[0] == i:
                        doc1_matched = True
                        break
                if doc1_matched:
                    continue
                if matched_doc2[j]:
                    continue
                filtered_pairs.append((i, j, sim))

            if filtered_pairs:
                print(f"  → {len(filtered_pairs)} candidates after filtering")

                # Limit concurrent calls to avoid overwhelming Ollama
                max_workers = 5

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all LLM tasks
                    future_to_pair = {
                        executor.submit(
                            self.compare_with_llm,
                            doc1_clauses[i]['text'],
                            doc2_clauses[j]['text']
                        ): (i, j, sim)
                        for i, j, sim in filtered_pairs
                    }

                    # Process results as they complete
                    for future in as_completed(future_to_pair):
                        i, j, sim = future_to_pair[future]
                        try:
                            result = future.result()
                            # Check if both clauses are still unmatched
                            doc1_matched = False
                            for pair in match_pairs:
                                if pair[0] == i:
                                    doc1_matched = True
                                    break

                            if not doc1_matched and not matched_doc2[j]:
                                if result.get('match', False) and result.get('confidence', 0) > 0.7:
                                    matched_doc2[j] = True
                                    match_pairs.append((i, j, sim, result.get('reason', 'LLM match')))
                                    llm_matches += 1
                                    print(f"  ✅ LLM matched: Doc1[{i}] ↔ Doc2[{j}] (sim: {sim:.3f})")
                        except Exception as e:
                            print(f"  ❌ LLM error for pair ({i}, {j}): {e}")

                print(f"  → {llm_matches} LLM matches found")

        # Build matching_details and only_in_doc1
        match_map = {i: (j, sim, reason) for i, j, sim, reason in match_pairs}

        matching_details = []
        only_in_doc1 = []

        for i, clause1 in enumerate(doc1_clauses):
            if i in match_map:
                j, sim, reason = match_map[i]
                clause2 = doc2_clauses[j]
                matching_details.append({
                    'clause_number': clause1.get('number', str(i+1)),
                    'clause_text': clause1['text'],
                    'found_match': True,
                    'best_match': {
                        'clause_idx': j,
                        'similarity': sim,
                        'confidence': 1.0 if 'high' in reason.lower() or 'mutual' in reason.lower() else 0.9,
                        'reason': reason,
                        'key_differences': [],
                        'used_llm': 'LLM' in reason
                    },
                    'top_similarity': sim,
                    'top_match_idx': j
                })
            else:
                # Unmatched doc1
                top_j = np.argmax(sim_matrix[i, :])
                top_sim = sim_matrix[i, top_j]
                matching_details.append({
                    'clause_number': clause1.get('number', str(i+1)),
                    'clause_text': clause1['text'],
                    'found_match': False,
                    'best_match': None,
                    'top_similarity': top_sim,
                    'top_match_idx': top_j
                })
                only_in_doc1.append({
                    'text': clause1['text'],
                    'number': clause1.get('number', str(i+1)),
                    'closest_match': doc2_clauses[top_j]['text'] if top_j < n2 else "",
                    'similarity': top_sim,
                    'metadata': clause1.get('metadata', {})
                })

        # Collect unmatched doc2 clauses
        only_in_doc2 = []
        for j in range(n2):
            if not matched_doc2[j]:
                sims = cosine_similarity([emb2[j]], emb1)[0]
                best_idx = np.argmax(sims)
                only_in_doc2.append({
                    'text': doc2_clauses[j]['text'],
                    'number': doc2_clauses[j].get('number', str(j+1)),
                    'closest_match': doc1_clauses[best_idx]['text'] if best_idx < n1 else "",
                    'similarity': sims[best_idx],
                    'metadata': doc2_clauses[j].get('metadata', {})
                })

        elapsed = time.time() - start

        # Compute counts
        llm_matches = sum(1 for _, _, _, reason in match_pairs if 'LLM' in reason)
        high_matches = len(match_pairs) - llm_matches

        print(f"✅ Final matches: {len(match_pairs)} total ({high_matches} via high-sim, {llm_matches} via LLM)")
        print(f"⏱️  Total time: {elapsed:.2f}s")

        return {
            'only_in_doc1': only_in_doc1,
            'only_in_doc2': only_in_doc2,
            'matching_details': matching_details,
            'total_doc1': n1,
            'total_doc2': n2,
            'matching_count': len(match_pairs),
            'processing_time': elapsed,
            'similarity_threshold': similarity_threshold,
            'high_similarity_threshold': high_similarity_threshold,
            'llm_matches': llm_matches,
            'high_sim_matches': high_matches,
            'doc2_best_similarities': [0.0] * n2
        }