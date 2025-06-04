// main.c
#include <stdio.h>
#include <stdint.h> // For size_t or uintptr_t if you want to be explicit, though usize often maps well.

// Include the generated header file.
// Make sure your compiler's include path points to `your_workspace_root/target/include/`
#include "strapdown.h" // Or "strapdown.h" if you named it that

int main() {
    size_t result = strapdown_add(15, 27);
    printf("Result from Rust (strapdown_add): %zu\n", result);
    return 0;
}