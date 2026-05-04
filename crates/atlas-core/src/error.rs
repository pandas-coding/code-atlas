use std::fmt;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorKind {
    Io,
    UnsupportedLanguage,
    Parse,
    InvalidInput,
    Internal,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ErrorContext {
    pub path: Option<PathBuf>,
    pub language: Option<crate::model::FileLanguage>,
    pub operation: Option<&'static str>,
}

impl ErrorContext {
    pub fn with_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.path = Some(path.into());
        self
    }

    pub fn with_language(mut self, language: crate::model::FileLanguage) -> Self {
        self.language = Some(language);
        self
    }

    pub fn with_operation(mut self, operation: &'static str) -> Self {
        self.operation = Some(operation);
        self
    }
}

#[derive(Debug, Clone)]
pub struct AtlasError {
    pub kind: ErrorKind,
    pub message: String,
    pub context: ErrorContext,
}

impl AtlasError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            context: ErrorContext::default(),
        }
    }

    pub fn with_context(mut self, context: ErrorContext) -> Self {
        self.context = context;
        self
    }

    pub fn io(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Io, message)
    }

    pub fn parse(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Parse, message)
    }

    pub fn unsupported_language(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::UnsupportedLanguage, message)
    }

    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::InvalidInput, message)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Internal, message)
    }
}

impl fmt::Display for AtlasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)?;

        if let Some(operation) = self.context.operation {
            write!(f, " [operation={operation}]")?;
        }

        if let Some(path) = &self.context.path {
            write!(f, " [path={}]", path.display())?;
        }

        if let Some(language) = self.context.language {
            write!(f, " [language={language}]")?;
        }

        Ok(())
    }
}

impl std::error::Error for AtlasError {}

impl From<std::io::Error> for AtlasError {
    fn from(value: std::io::Error) -> Self {
        AtlasError::io(value.to_string())
    }
}

pub type AtlasResult<T> = Result<T, AtlasError>;
