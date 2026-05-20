use crate::error::VdbResult;

pub trait EmbeddingService: Send + Sync {
    fn embed(&self, texts: &[&str]) -> VdbResult<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
}

pub struct MockEmbeddingService {
    dimension: usize,
}

impl MockEmbeddingService {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl EmbeddingService for MockEmbeddingService {
    fn embed(&self, texts: &[&str]) -> VdbResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            let mut vector = vec![0.0f32; self.dimension];
            if !text.is_empty() {
                let bytes = text.as_bytes();
                for (i, &byte) in bytes.iter().enumerate() {
                    if i >= self.dimension {
                        break;
                    }
                    vector[i] = (byte as f32) / 255.0;
                }
                let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for val in vector.iter_mut() {
                        *val /= norm;
                    }
                }
            }
            results.push(vector);
        }
        Ok(results)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_embedding_dimension() {
        for dim in [8, 32, 64, 128, 256] {
            let service = MockEmbeddingService::new(dim);
            assert_eq!(service.dimension(), dim);
            let results = service.embed(&["hello"]).unwrap();
            assert_eq!(results[0].len(), dim);
        }
    }

    #[test]
    fn test_mock_embedding_consistency_same_call() {
        let service = MockEmbeddingService::new(64);
        let r1 = service.embed(&["hello world"]).unwrap();
        let r2 = service.embed(&["hello world"]).unwrap();
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_mock_embedding_consistency_across_batches() {
        let service = MockEmbeddingService::new(32);
        let single = service.embed(&["alpha"]).unwrap()[0].clone();
        let batch = service.embed(&["alpha", "beta"]).unwrap();
        assert_eq!(single, batch[0]);
    }

    #[test]
    fn test_mock_embedding_different_inputs() {
        let service = MockEmbeddingService::new(64);
        let r1 = service.embed(&["hello"]).unwrap();
        let r2 = service.embed(&["world"]).unwrap();
        assert_ne!(r1[0], r2[0]);
    }

    #[test]
    fn test_mock_embedding_empty_text() {
        let service = MockEmbeddingService::new(32);
        let results = service.embed(&[""]).unwrap();
        assert_eq!(results[0].len(), 32);
        assert!(results[0].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_mock_embedding_batch_ordering() {
        let service = MockEmbeddingService::new(16);
        let results = service.embed(&["alpha", "beta", "gamma"]).unwrap();
        assert_eq!(results.len(), 3);
        let single_alpha = service.embed(&["alpha"]).unwrap()[0].clone();
        let single_beta = service.embed(&["beta"]).unwrap()[0].clone();
        assert_eq!(results[0], single_alpha);
        assert_eq!(results[1], single_beta);
    }

    #[test]
    fn test_mock_embedding_empty_batch() {
        let service = MockEmbeddingService::new(16);
        let results = service.embed(&[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_mock_embedding_normalized() {
        let service = MockEmbeddingService::new(32);
        let results = service.embed(&["some text here"]).unwrap();
        let norm: f32 = results[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "non-empty text embedding should be normalized");
    }
}
