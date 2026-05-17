mod config;
mod embedding;
mod error;
mod model;
mod onnx;
mod similarity;
mod store;

pub use config::EmbeddingConfig;
pub use embedding::EmbeddingService;
pub use error::{VdbError, VdbResult};
pub use model::{EmbeddingVector, SearchQuery, SearchResult};
pub use onnx::OnnxEmbeddingService;
pub use similarity::{cosine_similarity, dot_product};
pub use store::{InMemoryVectorStore, VectorStore};
