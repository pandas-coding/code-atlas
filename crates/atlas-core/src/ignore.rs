use std::path::Path;

/// Default directory names that should be ignored during scanning.
const IGNORED_DIRS: &[&str] = &[
    ".git",
    ".svn",
    ".hg",
    "target",
    "node_modules",
    ".next",
    ".nuxt",
    "dist",
    "build",
    "__pycache__",
    ".venv",
    "venv",
];

/// Checks if a directory should be ignored based on its name.
///
/// This function implements the standard ignore rules for common
/// directories that typically don't contain user source code.
pub fn should_ignore_dir(path: &Path) -> bool {
    if let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) {
        IGNORED_DIRS.contains(&dir_name)
    } else {
        false
    }
}

/// Checks if a file should be ignored based on its name/extension.
///
/// Filters out common non-source files like binaries, images, etc.
pub fn should_ignore_file(path: &Path) -> bool {
    let file_name = match path.file_name().and_then(|n| n.to_str()) {
        Some(name) => name,
        None => return true,
    };

    // Skip hidden files (starting with dot)
    if file_name.starts_with('.') {
        return true;
    }

    // Skip common non-source extensions
    let ignored_extensions = [
        // Binary/executable
        "exe", "dll", "so", "dylib", "o", "a", "lib", // Images
        "png", "jpg", "jpeg", "gif", "bmp", "ico", "svg", "webp", // Audio/Video
        "mp3", "mp4", "wav", "avi", "mov", // Documents
        "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", // Archives
        "zip", "tar", "gz", "rar", "7z", // Compiled files
        "pyc", "pyo", "class", "beam", // Lock files and other generated files
        "lock",
    ];

    if let Some(ext) = path.extension().and_then(|e| e.to_str())
        && ignored_extensions.contains(&ext.to_lowercase().as_str())
    {
        return true;
    }

    // Skip specific files
    let ignored_files = [
        ".DS_Store",
        "Thumbs.db",
        "Cargo.lock",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
    ];

    if ignored_files.contains(&file_name) {
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_ignore_git_dir() {
        assert!(should_ignore_dir(Path::new("/project/.git")));
        assert!(should_ignore_dir(Path::new(".git")));
    }

    #[test]
    fn test_should_ignore_target_dir() {
        assert!(should_ignore_dir(Path::new("/project/target")));
        assert!(should_ignore_dir(Path::new("target")));
    }

    #[test]
    fn test_should_ignore_node_modules() {
        assert!(should_ignore_dir(Path::new("/project/node_modules")));
        assert!(should_ignore_dir(Path::new("node_modules")));
    }

    #[test]
    fn test_should_not_ignore_normal_dirs() {
        assert!(!should_ignore_dir(Path::new("/project/src")));
        assert!(!should_ignore_dir(Path::new("crates")));
        assert!(!should_ignore_dir(Path::new("tests")));
    }

    #[test]
    fn test_should_ignore_binary_files() {
        assert!(should_ignore_file(Path::new("program.exe")));
        assert!(should_ignore_file(Path::new("lib.so")));
        assert!(should_ignore_file(Path::new("lib.dylib")));
    }

    #[test]
    fn test_should_ignore_image_files() {
        assert!(should_ignore_file(Path::new("image.png")));
        assert!(should_ignore_file(Path::new("photo.jpg")));
        assert!(should_ignore_file(Path::new("icon.svg")));
    }

    #[test]
    fn test_should_ignore_hidden_files() {
        assert!(should_ignore_file(Path::new(".gitignore")));
        assert!(should_ignore_file(Path::new(".env")));
    }

    #[test]
    fn test_should_ignore_lock_files() {
        assert!(should_ignore_file(Path::new("Cargo.lock")));
        assert!(should_ignore_file(Path::new("package-lock.json")));
        assert!(should_ignore_file(Path::new("yarn.lock")));
    }

    #[test]
    fn test_should_not_ignore_source_files() {
        assert!(!should_ignore_file(Path::new("main.rs")));
        assert!(!should_ignore_file(Path::new("index.js")));
        assert!(!should_ignore_file(Path::new("app.tsx")));
        assert!(!should_ignore_file(Path::new("lib.rs")));
    }
}
