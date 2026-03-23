from __future__ import annotations
import argparse
import os
from embedding_stage import build_embedding_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build embedding artifacts from translated chunk JSON")
    parser.add_argument(
        "--input",
        default="translated_chunks.json",
        help="Path to translated chunks JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default="embedding_artifacts",
        help="Directory to write embedding artifacts",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name (used when HuggingFace backend is available)",
    )

    args = parser.parse_args()

    result = build_embedding_artifacts(
        input_json_path=args.input,
        output_dir=args.output_dir,
        model_name=args.model,
    )

    print("=" * 80)
    print("EMBEDDING BUILD COMPLETE")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print(f"Backend: {result['meta']['embedding_backend']}")
    print(f"Model: {result['meta']['embedding_model']}")
    print(f"Chunks indexed: {result['meta']['chunk_count']}")
    print(f"Embedding dimension: {result['meta']['embedding_dim']}")
    print("-")
    print(f"Embeddings matrix: {result['embeddings_path']}")
    print(f"Normalized chunks: {result['chunks_path']}")
    print(f"Vector store records: {result['vector_records_path']}")
    print(f"Build metadata: {result['meta_path']}")


if __name__ == "__main__":
    main()
