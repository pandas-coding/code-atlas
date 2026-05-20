use atlas_core::{CodeChunk, IndexResult, SearchOptions, index_path, search_with_service};
use atlas_parser::parse_source;
use atlas_vdb::{
    EmbeddingConfig, EmbeddingService, EmbeddingVector, InMemoryVectorStore, MockEmbeddingService,
    VectorStore, cosine_similarity,
};
use std::path::PathBuf;
use tempfile::TempDir;

fn mock_embedding_config() -> EmbeddingConfig {
    EmbeddingConfig::new(PathBuf::from("/dummy/model.onnx"), 32, 512)
}

fn create_test_project_with_chunks() -> TempDir {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(root.join("src/main.rs"), "fn main() { }").unwrap();
    std::fs::write(root.join("src/lib.rs"), "pub fn hello() { }").unwrap();
    std::fs::write(root.join("src/utils.rs"), "fn helper() { }").unwrap();
    tmp
}

fn run_embedding_with_mock(
    index_result: &IndexResult,
    embedding_service: &dyn EmbeddingService,
    batch_size: usize,
    vector_store_path: &std::path::Path,
) -> (usize, usize, usize) {
    let all_chunks: Vec<&CodeChunk> = index_result
        .files
        .iter()
        .flat_map(|f| f.chunks.iter())
        .collect();

    if all_chunks.is_empty() {
        return (0, 0, embedding_service.dimension());
    }

    let batch_size = if batch_size == 0 { 32 } else { batch_size };
    let mut store = InMemoryVectorStore::new();
    let mut embedded = 0usize;
    let mut emb_errors = 0usize;

    for batch in all_chunks.chunks(batch_size) {
        let texts: Vec<&str> = batch.iter().map(|c| c.source_text.as_str()).collect();
        match embedding_service.embed(&texts) {
            Ok(vectors) => {
                let mut embedding_vectors = Vec::with_capacity(vectors.len());
                for (chunk, vector) in batch.iter().zip(vectors.into_iter()) {
                    embedding_vectors.push(EmbeddingVector::new(&chunk.id, vector));
                }
                match store.add(embedding_vectors) {
                    Ok(()) => embedded += batch.len(),
                    Err(_) => emb_errors += batch.len(),
                }
            }
            Err(_) => emb_errors += batch.len(),
        }
    }

    let dimension = embedding_service.dimension();

    if embedded > 0 {
        if let Some(parent) = vector_store_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        store.save(vector_store_path).unwrap();
    }

    (embedded, emb_errors, dimension)
}

#[test]
fn test_embedding_output_dimension() {
    for dim in [8, 16, 32, 64, 128] {
        let service = MockEmbeddingService::new(dim);
        assert_eq!(service.dimension(), dim);
        let results = service.embed(&["test input"]).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), dim);
    }
}

#[test]
fn test_embedding_consistency_same_input() {
    let service = MockEmbeddingService::new(32);
    let r1 = service.embed(&["hello world"]).unwrap();
    let r2 = service.embed(&["hello world"]).unwrap();
    assert_eq!(r1, r2);
}

#[test]
fn test_embedding_consistency_across_calls() {
    let service = MockEmbeddingService::new(32);
    let single = service.embed(&["alpha"]).unwrap()[0].clone();
    let batch = service.embed(&["alpha", "beta"]).unwrap();
    assert_eq!(single, batch[0]);
}

#[test]
fn test_embedding_consistency_different_inputs_differ() {
    let service = MockEmbeddingService::new(32);
    let r1 = service.embed(&["hello"]).unwrap();
    let r2 = service.embed(&["world"]).unwrap();
    assert_ne!(r1[0], r2[0]);
}

#[test]
fn test_vector_store_write_and_read() {
    let service = MockEmbeddingService::new(32);
    let texts = vec!["fn main() { }", "pub fn hello() { }", "fn helper() { }"];
    let vectors = service.embed(&texts).unwrap();

    let mut store = InMemoryVectorStore::new();
    let evs: Vec<EmbeddingVector> = texts
        .iter()
        .zip(vectors)
        .enumerate()
        .map(|(i, (_, vector))| EmbeddingVector::new(format!("chunk_{}", i), vector))
        .collect();
    store.add(evs).unwrap();

    assert_eq!(store.len(), 3);
    for i in 0..3 {
        let found = store.find(&format!("chunk_{}", i));
        assert!(found.is_some());
        assert_eq!(found.unwrap().vector.len(), 32);
    }
}

#[test]
fn test_vector_store_persistence_and_reload() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("test.vdb");

    let service = MockEmbeddingService::new(32);
    let texts = vec!["fn main() { }", "pub fn hello() { }"];
    let vectors = service.embed(&texts).unwrap();

    let mut store = InMemoryVectorStore::new();
    let evs: Vec<EmbeddingVector> = texts
        .iter()
        .zip(vectors)
        .enumerate()
        .map(|(i, (_, vector))| EmbeddingVector::new(format!("chunk_{}", i), vector))
        .collect();
    store.add(evs).unwrap();
    store.save(&path).unwrap();

    let loaded = InMemoryVectorStore::load(&path).unwrap();
    assert_eq!(loaded.len(), 2);
    for i in 0..2 {
        let original = store.find(&format!("chunk_{}", i)).unwrap();
        let reloaded = loaded.find(&format!("chunk_{}", i)).unwrap();
        assert_eq!(original.vector, reloaded.vector);
        assert_eq!(original.chunk_id, reloaded.chunk_id);
    }
}

#[test]
fn test_cosine_similarity_correctness() {
    let identical = vec![1.0, 0.0, 0.0];
    assert!((cosine_similarity(&identical, &identical) - 1.0).abs() < 1e-6);

    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-6);

    let c = vec![-1.0, 0.0, 0.0];
    assert!((cosine_similarity(&a, &c) - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_cosine_similarity_symmetry() {
    let pairs = vec![
        (vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]),
        (vec![1.0, 0.0], vec![0.0, 1.0]),
        (vec![3.0, 4.0], vec![1.0, 0.0]),
        (vec![1.0, -1.0, 0.0], vec![1.0, 1.0, 0.0]),
    ];
    for (a, b) in pairs {
        let ab = cosine_similarity(&a, &b);
        let ba = cosine_similarity(&b, &a);
        assert!((ab - ba).abs() < 1e-6, "Not symmetric: cos(a,b)={}, cos(b,a)={}", ab, ba);
    }
}

#[test]
fn test_cosine_similarity_scale_invariant() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let b_scaled: Vec<f32> = b.iter().map(|x| x * 10.0).collect();
    let s1 = cosine_similarity(&a, &b);
    let s2 = cosine_similarity(&a, &b_scaled);
    assert!((s1 - s2).abs() < 1e-6, "Scaling should not affect cosine similarity");
}

#[test]
fn test_top_k_search_correctness() {
    let service = MockEmbeddingService::new(32);
    let texts = vec!["alpha function", "beta method", "gamma trait"];
    let vectors = service.embed(&texts).unwrap();

    let mut store = InMemoryVectorStore::new();
    let evs: Vec<EmbeddingVector> = texts
        .iter()
        .zip(vectors)
        .enumerate()
        .map(|(i, (_, vector))| EmbeddingVector::new(format!("chunk_{}", i), vector))
        .collect();
    store.add(evs).unwrap();

    let query = service.embed(&["alpha function"]).unwrap();
    let results = store.search(&query[0], 2, None).unwrap();
    assert_eq!(results.len(), 2);
    assert!(results[0].1 >= results[1].1);
    assert_eq!(results[0].0, "chunk_0");
}

#[test]
fn test_top_k_search_with_min_score() {
    let service = MockEmbeddingService::new(32);
    let texts = vec!["alpha function", "beta method", "gamma trait"];
    let vectors = service.embed(&texts).unwrap();

    let mut store = InMemoryVectorStore::new();
    let evs: Vec<EmbeddingVector> = texts
        .iter()
        .zip(vectors)
        .enumerate()
        .map(|(i, (_, vector))| EmbeddingVector::new(format!("chunk_{}", i), vector))
        .collect();
    store.add(evs).unwrap();

    let query = service.embed(&["alpha function"]).unwrap();
    let results = store.search(&query[0], 10, Some(0.99)).unwrap();
    for (_, score) in &results {
        assert!(*score >= 0.99, "Score {} below min_score 0.99", score);
    }

    let all_results = store.search(&query[0], 10, None).unwrap();
    assert!(results.len() <= all_results.len());
}

#[test]
fn test_end_to_end_index_with_embedding() {
    let tmp = create_test_project_with_chunks();
    let index_result = index_path(tmp.path(), &parse_source).unwrap();

    let vdb_path = tmp.path().join("vectors.vdb");
    let service = MockEmbeddingService::new(32);
    let (embedded, errors, dimension) =
        run_embedding_with_mock(&index_result, &service, 32, &vdb_path);

    assert!(embedded > 0);
    assert_eq!(errors, 0);
    assert_eq!(dimension, 32);
    assert!(vdb_path.exists());

    let loaded = InMemoryVectorStore::load(&vdb_path).unwrap();
    assert_eq!(loaded.len(), index_result.stats.total_chunks);
}

#[test]
fn test_end_to_end_index_without_embedding_unchanged() {
    let tmp = create_test_project_with_chunks();
    let result = index_path(tmp.path(), &parse_source).unwrap();

    assert_eq!(result.stats.embedded_chunks, 0);
    assert_eq!(result.stats.embedding_errors, 0);
    assert_eq!(result.stats.embedding_dimension, 0);
    assert!(result.stats.total_chunks > 0);
}

#[test]
fn test_end_to_end_semantic_search() {
    let tmp = create_test_project_with_chunks();
    let index_result = index_path(tmp.path(), &parse_source).unwrap();

    let vdb_path = tmp.path().join("vectors.vdb");
    let service = MockEmbeddingService::new(32);
    run_embedding_with_mock(&index_result, &service, 32, &vdb_path);

    let search_opts = SearchOptions::new(mock_embedding_config(), &vdb_path);
    let results =
        search_with_service("fn main", tmp.path(), &parse_source, &search_opts, &service).unwrap();

    assert!(!results.is_empty());
    for result in &results {
        assert!(!result.chunk_id.is_empty());
        assert!(!result.chunk.source_text.is_empty());
    }
}

#[test]
fn test_end_to_end_semantic_search_with_top_k_limit() {
    let tmp = create_test_project_with_chunks();
    let index_result = index_path(tmp.path(), &parse_source).unwrap();

    let vdb_path = tmp.path().join("vectors.vdb");
    let service = MockEmbeddingService::new(32);
    run_embedding_with_mock(&index_result, &service, 32, &vdb_path);

    let search_opts = SearchOptions::new(mock_embedding_config(), &vdb_path).with_top_k(1);
    let results =
        search_with_service("fn main", tmp.path(), &parse_source, &search_opts, &service).unwrap();

    assert_eq!(results.len(), 1);
}

#[test]
fn test_end_to_end_search_empty_vector_store() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("main.rs"), "fn main() { }").unwrap();

    let vdb_path = tmp.path().join("empty.vdb");
    let store = InMemoryVectorStore::new();
    store.save(&vdb_path).unwrap();

    let service = MockEmbeddingService::new(32);
    let search_opts = SearchOptions::new(mock_embedding_config(), &vdb_path);
    let results =
        search_with_service("test", tmp.path(), &parse_source, &search_opts, &service).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_end_to_end_search_nonexistent_vector_store() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("main.rs"), "fn main() { }").unwrap();

    let vdb_path = tmp.path().join("nonexistent.vdb");
    let service = MockEmbeddingService::new(32);
    let search_opts = SearchOptions::new(mock_embedding_config(), &vdb_path);
    let result = search_with_service("test", tmp.path(), &parse_source, &search_opts, &service);
    assert!(result.is_err());
}

#[test]
fn test_embedding_batch_size_respected() {
    let tmp = create_test_project_with_chunks();
    let index_result = index_path(tmp.path(), &parse_source).unwrap();

    let service = MockEmbeddingService::new(32);
    let total_chunks = index_result.stats.total_chunks;

    for batch_size in [1, 2, 5, 10, 32, 100] {
        let vdb_path = tmp.path().join(format!("batch_{}.vdb", batch_size));
        let (embedded, errors, _) =
            run_embedding_with_mock(&index_result, &service, batch_size, &vdb_path);

        assert_eq!(errors, 0, "batch_size={} had errors", batch_size);
        assert_eq!(embedded, total_chunks, "batch_size={} embedded count mismatch", batch_size);
        assert!(vdb_path.exists(), "batch_size={} vector store not created", batch_size);

        let loaded = InMemoryVectorStore::load(&vdb_path).unwrap();
        assert_eq!(loaded.len(), total_chunks, "batch_size={} loaded count mismatch", batch_size);
    }
}

#[test]
fn test_embedding_with_no_chunks() {
    let tmp = TempDir::new().unwrap();
    let index_result = index_path(tmp.path(), &parse_source).unwrap();

    assert_eq!(index_result.stats.total_chunks, 0);

    let vdb_path = tmp.path().join("vectors.vdb");
    let service = MockEmbeddingService::new(32);
    let (embedded, errors, dimension) =
        run_embedding_with_mock(&index_result, &service, 32, &vdb_path);

    assert_eq!(embedded, 0);
    assert_eq!(errors, 0);
    assert_eq!(dimension, 32);
}
