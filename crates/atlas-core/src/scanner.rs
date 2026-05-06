use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::error::{AtlasError, AtlasResult};
use crate::ignore;
use crate::model::{FileLanguage, SourceFile};

/// Options for configuring the file scanner.
#[derive(Debug, Clone)]
pub struct ScanOptions {
    /// The root directory to scan.
    pub root: PathBuf,
    /// Maximum directory depth to scan. None means unlimited.
    pub max_depth: Option<usize>,
}

impl ScanOptions {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            max_depth: None,
        }
    }

    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }
}

/// Result of a directory scan operation.
#[derive(Debug, Clone)]
pub struct ScanResult {
    /// Files that were successfully identified as source files.
    pub source_files: Vec<PathBuf>,
    /// Files that were skipped (unsupported language, binary, etc.).
    pub skipped_files: Vec<PathBuf>,
    /// Errors encountered during scanning.
    pub errors: Vec<AtlasError>,
}

impl ScanResult {
    pub fn new() -> Self {
        Self {
            source_files: Vec::new(),
            skipped_files: Vec::new(),
            errors: Vec::new(),
        }
    }
}

impl Default for ScanResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Scanner that walks directories and identifies source files.
///
/// The scanner implements the following logic:
/// 1. Recursively walk through directories
/// 2. Skip ignored directories (`.git`, `target`, `node_modules`, etc.)
/// 3. Skip ignored files (binary, images, lock files, etc.)
/// 4. Identify source files by extension
/// 5. Normalize file paths
pub struct Scanner;

impl Scanner {
    /// Scans a directory and returns a list of source files that can be parsed.
    ///
    /// This is the main entry point for file discovery. It walks the directory
    /// tree, applying ignore rules and language detection.
    pub fn scan(options: &ScanOptions) -> ScanResult {
        let mut result = ScanResult::new();

        if !options.root.exists() {
            result.errors.push(
                AtlasError::io(format!(
                    "Root path does not exist: {}",
                    options.root.display()
                ))
            );
            return result;
        }

        if !options.root.is_dir() {
            // If the root is a single file, just check if it's a source file
            let normalized = Self::normalize_path(&options.root);
            if Self::is_source_file(&normalized) {
                result.source_files.push(normalized);
            } else {
                result.skipped_files.push(normalized);
            }
            return result;
        }

        Self::walk_directory(&options.root, 0, options.max_depth, &mut result);
        result
    }

    /// Recursively walks a directory, collecting source files.
    fn walk_directory(
        dir: &Path,
        current_depth: usize,
        max_depth: Option<usize>,
        result: &mut ScanResult,
    ) {
        // Check depth limit
        if let Some(max) = max_depth
            && current_depth > max
        {
            return;
        }

        // Read directory entries
        let entries = match fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(e) => {
                result.errors.push(
                    AtlasError::io(format!(
                        "Failed to read directory {}: {e}",
                        dir.display()
                    ))
                );
                return;
            }
        };

        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(e) => {
                    result.errors.push(
                        AtlasError::io(format!(
                            "Failed to read entry in {}: {e}",
                            dir.display()
                        ))
                    );
                    continue;
                }
            };

            let path = entry.path();
            let file_type = match entry.file_type() {
                Ok(ft) => ft,
                Err(e) => {
                    result.errors.push(
                        AtlasError::io(format!(
                            "Failed to get file type for {}: {e}",
                            path.display()
                        ))
                    );
                    continue;
                }
            };

            if file_type.is_dir() {
                // Check if directory should be ignored
                if ignore::should_ignore_dir(&path) {
                    continue;
                }
                // Recurse into subdirectory
                Self::walk_directory(&path, current_depth + 1, max_depth, result);
            } else if file_type.is_file() {
                // Normalize the path
                let normalized = Self::normalize_path(&path);

                // Check if file should be ignored
                if ignore::should_ignore_file(&normalized) {
                    result.skipped_files.push(normalized);
                    continue;
                }

                // Check if file has a supported source extension
                if Self::is_source_file(&normalized) {
                    result.source_files.push(normalized);
                } else {
                    result.skipped_files.push(normalized);
                }
            }
            // Skip symlinks and other special files
        }
    }

    /// Checks if a file is a source file based on its extension.
    ///
    /// Returns true if the file has a recognized source code extension.
    pub fn is_source_file(path: &Path) -> bool {
        let language = FileLanguage::from_path(path);
        language.is_supported()
    }

    /// Normalizes a file path for consistent representation.
    ///
    /// This handles:
    /// - Converting to canonical path when possible
    /// - Resolving `.` and `..` components as a fallback
    /// - Standardizing path separators
    pub fn normalize_path(path: &Path) -> PathBuf {
        // Try to canonicalize the path, but fall back to manual cleanup if it fails
        // (e.g., during testing or for paths that don't exist yet)
        path.canonicalize().unwrap_or_else(|_| Self::clean_path(path))
    }

    /// Cleans a path by resolving `.` and `..` components without requiring
    /// the path to exist on disk.
    fn clean_path(path: &Path) -> PathBuf {
        let mut components = Vec::new();
        for component in path.components() {
            match component {
                Component::CurDir => {
                    // Skip `.` components
                }
                Component::ParentDir => {
                    // Pop the last non-root component if possible
                    let should_pop = match components.last() {
                        Some(c) => !matches!(c, Component::RootDir | Component::Prefix(_)),
                        None => false,
                    };
                    if should_pop {
                        components.pop();
                    } else {
                        components.push(component);
                    }
                }
                other => components.push(other),
            }
        }

        let mut result = PathBuf::new();
        for component in components {
            result.push(component);
        }
        result
    }

    /// Scans a directory and reads all source files into SourceFile objects.
    ///
    /// This is a convenience method that combines scanning with file reading.
    pub fn scan_and_read(options: &ScanOptions) -> Vec<AtlasResult<SourceFile>> {
        let scan_result = Self::scan(options);

        scan_result
            .source_files
            .into_iter()
            .map(|path| {
                let source_text = fs::read_to_string(&path).map_err(|e| {
                    AtlasError::io(format!(
                        "Failed to read file {}: {e}",
                        path.display()
                    ))
                })?;
                Ok(SourceFile::new(path, source_text))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_structure() -> TempDir {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        // Create some source files
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();
        fs::write(root.join("src/lib.rs"), "pub fn hello() {}").unwrap();
        fs::write(root.join("src/index.js"), "console.log('hello')").unwrap();
        fs::write(root.join("src/app.tsx"), "export default function App() {}").unwrap();

        // Create some ignored directories
        fs::create_dir_all(root.join(".git/objects")).unwrap();
        fs::write(root.join(".git/config"), "").unwrap();
        fs::create_dir_all(root.join("target/debug")).unwrap();
        fs::write(root.join("target/debug/program"), "").unwrap();
        fs::create_dir_all(root.join("node_modules/package")).unwrap();
        fs::write(root.join("node_modules/package/index.js"), "").unwrap();

        // Create some ignored files
        fs::write(root.join("image.png"), "").unwrap();
        fs::write(root.join(".gitignore"), "").unwrap();
        fs::write(root.join("Cargo.lock"), "").unwrap();

        tmp
    }

    #[test]
    fn test_scan_finds_source_files() {
        let tmp = create_test_structure();
        let options = ScanOptions::new(tmp.path());
        let result = Scanner::scan(&options);

        assert_eq!(result.source_files.len(), 4);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_scan_skips_ignored_dirs() {
        let tmp = create_test_structure();
        let options = ScanOptions::new(tmp.path());
        let result = Scanner::scan(&options);

        // Verify no files from ignored directories are included
        for file in &result.source_files {
            let path_str = file.to_string_lossy();
            assert!(!path_str.contains(".git"));
            assert!(!path_str.contains("target"));
            assert!(!path_str.contains("node_modules"));
        }
    }

    #[test]
    fn test_scan_skips_ignored_files() {
        let tmp = create_test_structure();
        let options = ScanOptions::new(tmp.path());
        let result = Scanner::scan(&options);

        // Verify ignored files are in skipped_files, not source_files
        for file in &result.skipped_files {
            let path_str = file.to_string_lossy();
            assert!(
                path_str.contains("image.png")
                    || path_str.contains(".gitignore")
                    || path_str.contains("Cargo.lock")
            );
        }
    }

    #[test]
    fn test_is_source_file() {
        assert!(Scanner::is_source_file(Path::new("main.rs")));
        assert!(Scanner::is_source_file(Path::new("index.js")));
        assert!(Scanner::is_source_file(Path::new("app.tsx")));
        assert!(!Scanner::is_source_file(Path::new("image.png")));
        assert!(!Scanner::is_source_file(Path::new("README.md")));
    }

    #[test]
    fn test_normalize_path_clean() {
        // Test the clean_path fallback (for non-existent paths)
        let path = Path::new("./src/../src/main.rs");
        let cleaned = Scanner::clean_path(path);
        assert_eq!(cleaned, PathBuf::from("src/main.rs"));
    }

    #[test]
    fn test_normalize_path() {
        let path = Path::new("./src/../src/main.rs");
        let normalized = Scanner::normalize_path(path);
        let normalized_str = normalized.to_string_lossy();
        // The normalized path should be cleaned up (no `..` components)
        assert!(
            !normalized_str.contains("/../"),
            "path still contains /../: {normalized_str}"
        );
        assert!(
            !normalized_str.ends_with(".."),
            "path ends with '..': {normalized_str}"
        );
    }

    #[test]
    fn test_max_depth() {
        let tmp = create_test_structure();
        let options = ScanOptions::new(tmp.path()).with_max_depth(0);
        let result = Scanner::scan(&options);

        // With max_depth=0, we should only get files in the root directory
        // (none in this test case, since all source files are in subdirectories)
        assert_eq!(result.source_files.len(), 0);
    }

    #[test]
    fn test_scan_single_file() {
        let tmp = create_test_structure();
        let file_path = tmp.path().join("src/main.rs");
        let options = ScanOptions::new(&file_path);
        let result = Scanner::scan(&options);

        assert_eq!(result.source_files.len(), 1);
        // Both sides are normalized via the same function to handle
        // platform-specific canonicalization (e.g., Windows UNC paths)
        let expected = Scanner::normalize_path(&file_path);
        assert_eq!(result.source_files[0], expected);
    }
}
