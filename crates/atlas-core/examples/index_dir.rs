use atlas_core::index_path;
use atlas_parser::parse_source;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "index_dir", about = "Index a directory or file with code-atlas")]
struct Args {
    #[arg(default_value = ".")]
    path: PathBuf,
}

fn main() {
    let cli = Args::parse();

    match index_path(&cli.path, &parse_source) {
        Ok(result) => {
            println!("Index result for: {}", cli.path.display());
            println!(
                "  files: {} parsed, {} skipped, {} total",
                result.stats.parsed_files,
                result.stats.skipped_files,
                result.stats.total_files,
            );
            println!("  chunks: {}", result.stats.total_chunks);
            println!("  errors: {}", result.stats.total_errors);

            for file_result in &result.files {
                println!();
                println!(
                    "  ─── {} ({}) ───",
                    file_result.file.path.display(),
                    file_result.file.language,
                );
                for chunk in &file_result.chunks {
                    println!(
                        "    {} `{}` lines {}-{}",
                        chunk.kind,
                        chunk.symbol_name.as_deref().unwrap_or("(anonymous)"),
                        chunk.span.start_line + 1,
                        chunk.span.end_line + 1,
                    );
                }
            }

            if !result.errors.is_empty() {
                println!();
                println!("  errors:");
                for e in &result.errors {
                    println!("    - {e}");
                }
            }
        }
        Err(e) => eprintln!("error: {e}"),
    }
}
