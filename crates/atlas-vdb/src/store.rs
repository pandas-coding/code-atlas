use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::error::{VdbContext, VdbError, VdbResult};
use crate::model::EmbeddingVector;
use crate::similarity::cosine_similarity;

const MAGIC: [u8; 4] = *b"ATVS";
const VERSION: u32 = 1;

pub trait VectorStore: Send + Sync {
    fn add(&mut self, vectors: Vec<EmbeddingVector>) -> VdbResult<()>;
    fn search(
        &self,
        query: &[f32],
        top_k: usize,
        min_score: Option<f32>,
    ) -> VdbResult<Vec<(String, f32)>>;
    fn save(&self, path: &Path) -> VdbResult<()>;
    fn load(path: &Path) -> VdbResult<Self>
    where
        Self: Sized;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn clear(&mut self);
    fn find(&self, chunk_id: &str) -> Option<&EmbeddingVector>;
}

#[derive(Debug)]
pub struct InMemoryVectorStore {
    vectors: Vec<EmbeddingVector>,
    index: HashMap<String, usize>,
    dimension: usize,
}

impl InMemoryVectorStore {
    pub fn new() -> Self {
        Self { vectors: Vec::new(), index: HashMap::new(), dimension: 0 }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            vectors: Vec::with_capacity(capacity),
            index: HashMap::with_capacity(capacity),
            dimension: 0,
        }
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

impl Default for InMemoryVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorStore for InMemoryVectorStore {
    fn add(&mut self, vectors: Vec<EmbeddingVector>) -> VdbResult<()> {
        for vector in vectors {
            if self.dimension == 0 {
                self.dimension = vector.dimension;
            } else if vector.dimension != self.dimension {
                return Err(VdbError::invalid_input(format!(
                    "Dimension mismatch: expected {}, got {}",
                    self.dimension, vector.dimension
                )));
            }

            if vector.vector.len() != vector.dimension {
                return Err(VdbError::invalid_input(format!(
                    "Vector length {} does not match declared dimension {}",
                    vector.vector.len(),
                    vector.dimension
                )));
            }

            if let Some(&existing_idx) = self.index.get(&vector.chunk_id) {
                self.vectors[existing_idx] = vector;
            } else {
                let idx = self.vectors.len();
                self.index.insert(vector.chunk_id.clone(), idx);
                self.vectors.push(vector);
            }
        }
        Ok(())
    }

    fn search(
        &self,
        query: &[f32],
        top_k: usize,
        min_score: Option<f32>,
    ) -> VdbResult<Vec<(String, f32)>> {
        if self.vectors.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        if query.len() != self.dimension {
            return Err(VdbError::invalid_input(format!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimension,
                query.len()
            )));
        }

        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        if query_norm == 0.0 {
            return Ok(Vec::new());
        }

        let min = min_score.unwrap_or(f32::NEG_INFINITY);

        let mut scores: Vec<(String, f32)> = self
            .vectors
            .iter()
            .map(|ev| {
                let score = cosine_similarity(query, &ev.vector);
                (ev.chunk_id.clone(), score)
            })
            .filter(|(_, score)| *score >= min)
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);

        Ok(scores)
    }

    fn save(&self, path: &Path) -> VdbResult<()> {
        let file = std::fs::File::create(path).map_err(|e| {
            VdbError::io(format!("Failed to create file: {}", path.display()))
                .with_context(VdbContext::default().with_path(path).with_operation("save"))
                .with_source(e.to_string())
        })?;

        let mut writer = BufWriter::new(file);

        writer
            .write_all(&MAGIC)
            .map_err(|e| VdbError::io("Failed to write magic bytes").with_source(e.to_string()))?;

        writer
            .write_all(&VERSION.to_le_bytes())
            .map_err(|e| VdbError::io("Failed to write version").with_source(e.to_string()))?;

        let count = self.vectors.len() as u32;
        writer
            .write_all(&count.to_le_bytes())
            .map_err(|e| VdbError::io("Failed to write count").with_source(e.to_string()))?;

        let dim = self.dimension as u32;
        writer
            .write_all(&dim.to_le_bytes())
            .map_err(|e| VdbError::io("Failed to write dimension").with_source(e.to_string()))?;

        for ev in &self.vectors {
            let chunk_id_bytes = ev.chunk_id.as_bytes();
            let chunk_id_len = chunk_id_bytes.len() as u32;
            writer.write_all(&chunk_id_len.to_le_bytes()).map_err(|e| {
                VdbError::io("Failed to write chunk_id length").with_source(e.to_string())
            })?;
            writer
                .write_all(chunk_id_bytes)
                .map_err(|e| VdbError::io("Failed to write chunk_id").with_source(e.to_string()))?;

            let vector_bytes: Vec<u8> = ev.vector.iter().flat_map(|f| f.to_le_bytes()).collect();
            writer.write_all(&vector_bytes).map_err(|e| {
                VdbError::io("Failed to write vector data").with_source(e.to_string())
            })?;
        }

        writer
            .flush()
            .map_err(|e| VdbError::io("Failed to flush file").with_source(e.to_string()))?;

        Ok(())
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn clear(&mut self) {
        self.vectors.clear();
        self.index.clear();
        self.dimension = 0;
    }

    fn load(path: &Path) -> VdbResult<Self> {
        if !path.exists() {
            return Err(VdbError::io(format!("File not found: {}", path.display()))
                .with_context(VdbContext::default().with_path(path).with_operation("load")));
        }

        let file = std::fs::File::open(path).map_err(|e| {
            VdbError::io(format!("Failed to open file: {}", path.display()))
                .with_context(VdbContext::default().with_path(path).with_operation("load"))
                .with_source(e.to_string())
        })?;

        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).map_err(|e| {
            VdbError::storage("Failed to read magic bytes")
                .with_context(VdbContext::default().with_path(path).with_operation("load"))
                .with_source(e.to_string())
        })?;

        if magic != MAGIC {
            return Err(VdbError::storage(format!(
                "Invalid file format: expected magic {:?}, got {:?}",
                std::str::from_utf8(&MAGIC).unwrap_or("????"),
                std::str::from_utf8(&magic).unwrap_or("????"),
            ))
            .with_context(VdbContext::default().with_path(path).with_operation("load")));
        }

        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes).map_err(|e| {
            VdbError::storage("Failed to read version")
                .with_context(VdbContext::default().with_path(path).with_operation("load"))
                .with_source(e.to_string())
        })?;
        let version = u32::from_le_bytes(version_bytes);

        if version != VERSION {
            return Err(VdbError::storage(format!(
                "Unsupported version: expected {}, got {}",
                VERSION, version
            ))
            .with_context(VdbContext::default().with_path(path).with_operation("load")));
        }

        let mut count_bytes = [0u8; 4];
        reader.read_exact(&mut count_bytes).map_err(|e| {
            VdbError::storage("Failed to read count")
                .with_context(VdbContext::default().with_path(path).with_operation("load"))
                .with_source(e.to_string())
        })?;
        let count = u32::from_le_bytes(count_bytes) as usize;

        let mut dim_bytes = [0u8; 4];
        reader.read_exact(&mut dim_bytes).map_err(|e| {
            VdbError::storage("Failed to read dimension")
                .with_context(VdbContext::default().with_path(path).with_operation("load"))
                .with_source(e.to_string())
        })?;
        let dimension = u32::from_le_bytes(dim_bytes) as usize;

        let mut store = InMemoryVectorStore {
            vectors: Vec::with_capacity(count),
            index: HashMap::with_capacity(count),
            dimension,
        };

        for _ in 0..count {
            let mut chunk_id_len_bytes = [0u8; 4];
            reader.read_exact(&mut chunk_id_len_bytes).map_err(|e| {
                VdbError::storage("Failed to read chunk_id length")
                    .with_context(VdbContext::default().with_path(path).with_operation("load"))
                    .with_source(e.to_string())
            })?;
            let chunk_id_len = u32::from_le_bytes(chunk_id_len_bytes) as usize;

            let mut chunk_id_bytes = vec![0u8; chunk_id_len];
            reader.read_exact(&mut chunk_id_bytes).map_err(|e| {
                VdbError::storage("Failed to read chunk_id")
                    .with_context(VdbContext::default().with_path(path).with_operation("load"))
                    .with_source(e.to_string())
            })?;

            let chunk_id = String::from_utf8(chunk_id_bytes).map_err(|e| {
                VdbError::storage(format!("Invalid chunk_id UTF-8: {e}"))
                    .with_context(VdbContext::default().with_path(path).with_operation("load"))
            })?;

            let mut vector_bytes = vec![0u8; dimension * 4];
            reader.read_exact(&mut vector_bytes).map_err(|e| {
                VdbError::storage("Failed to read vector data")
                    .with_context(VdbContext::default().with_path(path).with_operation("load"))
                    .with_source(e.to_string())
            })?;

            let vector: Vec<f32> = vector_bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            let ev = EmbeddingVector::new(chunk_id.clone(), vector);
            let idx = store.vectors.len();
            store.index.insert(chunk_id, idx);
            store.vectors.push(ev);
        }

        Ok(store)
    }

    fn find(&self, chunk_id: &str) -> Option<&EmbeddingVector> {
        self.index.get(chunk_id).map(|&i| &self.vectors[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::EmbeddingVector;

    fn make_vector(id: &str, values: Vec<f32>) -> EmbeddingVector {
        EmbeddingVector::new(id, values)
    }

    #[test]
    fn test_add_and_len() {
        let mut store = InMemoryVectorStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);

        store
            .add(vec![make_vector("c1", vec![1.0, 0.0, 0.0])])
            .unwrap();
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());

        store
            .add(vec![
                make_vector("c2", vec![0.0, 1.0, 0.0]),
                make_vector("c3", vec![0.0, 0.0, 1.0]),
            ])
            .unwrap();
        assert_eq!(store.len(), 3);
    }

    #[test]
    fn test_add_dimension_mismatch() {
        let mut store = InMemoryVectorStore::new();
        store.add(vec![make_vector("c1", vec![1.0, 0.0])]).unwrap();

        let result = store.add(vec![make_vector("c2", vec![1.0, 0.0, 0.0])]);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_upsert_by_chunk_id() {
        let mut store = InMemoryVectorStore::new();
        store.add(vec![make_vector("c1", vec![1.0, 0.0])]).unwrap();
        assert_eq!(store.len(), 1);

        store.add(vec![make_vector("c1", vec![0.0, 1.0])]).unwrap();
        assert_eq!(store.len(), 1);

        let found = store.find("c1").unwrap();
        assert_eq!(found.vector, vec![0.0, 1.0]);
    }

    #[test]
    fn test_find() {
        let mut store = InMemoryVectorStore::new();
        store
            .add(vec![make_vector("c1", vec![1.0, 0.0]), make_vector("c2", vec![0.0, 1.0])])
            .unwrap();

        assert!(store.find("c1").is_some());
        assert!(store.find("c2").is_some());
        assert!(store.find("c3").is_none());

        let found = store.find("c1").unwrap();
        assert_eq!(found.chunk_id, "c1");
        assert_eq!(found.vector, vec![1.0, 0.0]);
    }

    #[test]
    fn test_clear() {
        let mut store = InMemoryVectorStore::new();
        store
            .add(vec![make_vector("c1", vec![1.0, 0.0]), make_vector("c2", vec![0.0, 1.0])])
            .unwrap();
        assert_eq!(store.len(), 2);

        store.clear();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert_eq!(store.dimension(), 0);
    }

    #[test]
    fn test_search_top_k() {
        let mut store = InMemoryVectorStore::new();
        store
            .add(vec![
                make_vector("c1", vec![1.0, 0.0, 0.0]),
                make_vector("c2", vec![0.0, 1.0, 0.0]),
                make_vector("c3", vec![0.9, 0.1, 0.0]),
            ])
            .unwrap();

        let results = store.search(&[1.0, 0.0, 0.0], 2, None).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "c1");
        assert!(results[0].1 > results[1].1);
    }

    #[test]
    fn test_search_empty_store() {
        let store = InMemoryVectorStore::new();
        let results = store.search(&[1.0, 0.0], 5, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_query_dimension_mismatch() {
        let mut store = InMemoryVectorStore::new();
        store.add(vec![make_vector("c1", vec![1.0, 0.0])]).unwrap();

        let result = store.search(&[1.0, 0.0, 0.0], 5, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_search_min_score_filter() {
        let mut store = InMemoryVectorStore::new();
        store
            .add(vec![
                make_vector("c1", vec![1.0, 0.0, 0.0]),
                make_vector("c2", vec![0.0, 1.0, 0.0]),
                make_vector("c3", vec![0.9, 0.1, 0.0]),
            ])
            .unwrap();

        let results = store.search(&[1.0, 0.0, 0.0], 10, Some(0.9)).unwrap();
        for (_, score) in &results {
            assert!(*score >= 0.9, "Score {} below threshold 0.9", score);
        }
    }

    #[test]
    fn test_search_min_score_filters_all() {
        let mut store = InMemoryVectorStore::new();
        store
            .add(vec![
                make_vector("c1", vec![0.0, 1.0, 0.0]),
                make_vector("c2", vec![0.0, 0.0, 1.0]),
            ])
            .unwrap();

        let results = store.search(&[1.0, 0.0, 0.0], 10, Some(0.5)).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_results_sorted_descending() {
        let mut store = InMemoryVectorStore::new();
        store
            .add(vec![
                make_vector("c1", vec![1.0, 0.0, 0.0]),
                make_vector("c2", vec![0.0, 1.0, 0.0]),
                make_vector(
                    "c3",
                    vec![std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2, 0.0],
                ),
            ])
            .unwrap();

        let results = store.search(&[1.0, 0.0, 0.0], 10, None).unwrap();
        for i in 1..results.len() {
            assert!(
                results[i - 1].1 >= results[i].1,
                "Results not sorted descending: {} > {}",
                results[i - 1].1,
                results[i].1
            );
        }
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let dir = std::env::temp_dir().join("atlas_vdb_test_save_load");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.vdb");

        let mut store = InMemoryVectorStore::new();
        store
            .add(vec![
                make_vector("c1", vec![1.0, 2.0, 3.0]),
                make_vector("c2", vec![4.0, 5.0, 6.0]),
                make_vector("c3", vec![7.0, 8.0, 9.0]),
            ])
            .unwrap();

        store.save(&path).unwrap();

        let loaded = InMemoryVectorStore::load(&path).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.dimension(), 3);

        let v1 = loaded.find("c1").unwrap();
        assert_eq!(v1.vector, vec![1.0, 2.0, 3.0]);

        let v2 = loaded.find("c2").unwrap();
        assert_eq!(v2.vector, vec![4.0, 5.0, 6.0]);

        let v3 = loaded.find("c3").unwrap();
        assert_eq!(v3.vector, vec![7.0, 8.0, 9.0]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_file_not_found() {
        let result = InMemoryVectorStore::load(Path::new("/nonexistent/path/test.vdb"));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("File not found"));
    }

    #[test]
    fn test_load_invalid_format() {
        let dir = std::env::temp_dir().join("atlas_vdb_test_invalid");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.vdb");

        std::fs::write(&path, b"NOT_A_VALID_FILE").unwrap();

        let result = InMemoryVectorStore::load(&path);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("Invalid file format"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_add_batch_preserves_order() {
        let mut store = InMemoryVectorStore::new();
        let vectors: Vec<EmbeddingVector> = (0..5)
            .map(|i| make_vector(&format!("c{}", i), vec![i as f32, 0.0, 0.0]))
            .collect();
        store.add(vectors).unwrap();
        assert_eq!(store.len(), 5);
        for i in 0..5 {
            let found = store.find(&format!("c{}", i)).unwrap();
            assert_eq!(found.vector, vec![i as f32, 0.0, 0.0]);
        }
    }

    #[test]
    fn test_add_and_find_roundtrip() {
        let mut store = InMemoryVectorStore::new();
        let v1 = make_vector("chunk_a", vec![1.0, 2.0, 3.0]);
        let v2 = make_vector("chunk_b", vec![4.0, 5.0, 6.0]);
        store.add(vec![v1.clone(), v2.clone()]).unwrap();

        let found_a = store.find("chunk_a").unwrap();
        assert_eq!(found_a.chunk_id, "chunk_a");
        assert_eq!(found_a.vector, vec![1.0, 2.0, 3.0]);
        assert_eq!(found_a.dimension, 3);

        let found_b = store.find("chunk_b").unwrap();
        assert_eq!(found_b.chunk_id, "chunk_b");
        assert_eq!(found_b.vector, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_save_load_preserves_many_vectors() {
        let dir = std::env::temp_dir().join("atlas_vdb_test_many_vectors");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("many.vdb");

        let mut store = InMemoryVectorStore::new();
        let dim = 16;
        let count = 50;
        let mut vectors = Vec::with_capacity(count);
        for i in 0..count {
            let vals: Vec<f32> = (0..dim).map(|j| (i * dim + j) as f32 * 0.01).collect();
            vectors.push(make_vector(&format!("vec_{}", i), vals));
        }
        store.add(vectors).unwrap();
        assert_eq!(store.len(), count);

        store.save(&path).unwrap();
        let loaded = InMemoryVectorStore::load(&path).unwrap();
        assert_eq!(loaded.len(), count);
        assert_eq!(loaded.dimension(), dim);

        for i in 0..count {
            let found = loaded.find(&format!("vec_{}", i)).unwrap();
            let expected: Vec<f32> = (0..dim).map(|j| (i * dim + j) as f32 * 0.01).collect();
            assert_eq!(found.vector, expected);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_load_with_unicode_chunk_id() {
        let dir = std::env::temp_dir().join("atlas_vdb_test_unicode");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("unicode.vdb");

        let mut store = InMemoryVectorStore::new();
        store
            .add(vec![
                make_vector("模块::函数", vec![1.0, 0.0]),
                make_vector("クラス::メソッド", vec![0.0, 1.0]),
            ])
            .unwrap();

        store.save(&path).unwrap();
        let loaded = InMemoryVectorStore::load(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert!(loaded.find("模块::函数").is_some());
        assert!(loaded.find("クラス::メソッド").is_some());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_load_preserves_search_results() {
        let dir = std::env::temp_dir().join("atlas_vdb_test_search_after_load");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("search.vdb");

        let mut store = InMemoryVectorStore::new();
        store
            .add(vec![
                make_vector("c1", vec![1.0, 0.0, 0.0]),
                make_vector("c2", vec![0.0, 1.0, 0.0]),
                make_vector("c3", vec![0.9, 0.1, 0.0]),
            ])
            .unwrap();

        let before = store.search(&[1.0, 0.0, 0.0], 3, None).unwrap();

        store.save(&path).unwrap();
        let loaded = InMemoryVectorStore::load(&path).unwrap();
        let after = loaded.search(&[1.0, 0.0, 0.0], 3, None).unwrap();

        assert_eq!(before.len(), after.len());
        for (b, a) in before.iter().zip(after.iter()) {
            assert_eq!(b.0, a.0);
            assert!((b.1 - a.1).abs() < 1e-6);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_search_top_k_fewer_than_total() {
        let mut store = InMemoryVectorStore::new();
        store
            .add(vec![
                make_vector("c1", vec![1.0, 0.0]),
                make_vector("c2", vec![0.9, 0.1]),
                make_vector("c3", vec![0.0, 1.0]),
                make_vector("c4", vec![0.1, 0.9]),
            ])
            .unwrap();

        let results = store.search(&[1.0, 0.0], 2, None).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "c1");
    }

    #[test]
    fn test_search_top_k_greater_than_total() {
        let mut store = InMemoryVectorStore::new();
        store
            .add(vec![make_vector("c1", vec![1.0, 0.0]), make_vector("c2", vec![0.0, 1.0])])
            .unwrap();

        let results = store.search(&[1.0, 0.0], 10, None).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_top_k_zero() {
        let mut store = InMemoryVectorStore::new();
        store.add(vec![make_vector("c1", vec![1.0, 0.0])]).unwrap();
        let results = store.search(&[1.0, 0.0], 0, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_with_capacity() {
        let store = InMemoryVectorStore::with_capacity(100);
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert_eq!(store.dimension(), 0);
    }

    #[test]
    fn test_add_vector_length_mismatch_with_dimension() {
        let mut store = InMemoryVectorStore::new();
        let bad_vector =
            EmbeddingVector { chunk_id: "c1".to_string(), vector: vec![1.0, 0.0], dimension: 3 };
        let result = store.add(vec![bad_vector]);
        assert!(result.is_err());
    }
}
