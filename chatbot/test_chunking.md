# Test Document for Chunking

This is a test markdown document to verify the document chunking pipeline.

## Section 1: Introduction

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

## Section 2: Technical Details

This section contains some technical information about the chunking process:

1. Documents are read from the file system
2. The text is normalized for whitespace and encoding
3. Chunks are created using hierarchical separators
4. Each chunk is assigned a unique hash for deduplication
5. Chunks are saved to JSONL format for processing

## Section 3: Conclusion

The chunking process should produce multiple chunks from this document due to the paragraph structure and length of the content. Each chunk should have proper metadata including start position, end position, and a hash value.
