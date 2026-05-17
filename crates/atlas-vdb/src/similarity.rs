pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same dimension");
    let dot = dot_product(a, b);
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same dimension");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 0.0, 0.0];
        let score = cosine_similarity(&v, &v);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let score = cosine_similarity(&a, &b);
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let score = cosine_similarity(&a, &b);
        assert!((score - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_range() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let score = cosine_similarity(&a, &b);
        assert!((-1.0..=1.0).contains(&score));
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0];
        let score = cosine_similarity(&a, &b);
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_both_zero() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0];
        let score = cosine_similarity(&a, &b);
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_general() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dot = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0;
        let norm_a = (1.0 + 4.0 + 9.0_f32).sqrt();
        let norm_b = (16.0 + 25.0 + 36.0_f32).sqrt();
        let expected = dot / (norm_a * norm_b);
        let score = cosine_similarity(&a, &b);
        assert!((score - expected).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product(&a, &b);
        assert!((result - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let result = dot_product(&a, &b);
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_zero() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = dot_product(&a, &b);
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_negative_components() {
        let a = vec![1.0, -1.0, 0.0];
        let b = vec![1.0, 1.0, 0.0];
        let score = cosine_similarity(&a, &b);
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_all_negative_range() {
        let vectors = vec![
            (vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]),
            (vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]),
            (vec![1.0, 0.0, 0.0], vec![-1.0, 0.0, 0.0]),
            (vec![1.0, 1.0, 0.0], vec![-1.0, -1.0, 0.0]),
            (vec![3.0, 4.0], vec![4.0, 3.0]),
        ];
        for (a, b) in vectors {
            let score = cosine_similarity(&a, &b);
            assert!(
                (-1.0 - 1e-6..=1.0 + 1e-6).contains(&score),
                "Score {score} out of range for {a:?} vs {b:?}"
            );
        }
    }
}
