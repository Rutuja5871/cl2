# Import libraries
import os

# Read the DNA sequence
with open("datasets/dna_sequence.txt") as file:
    dna_sequence = file.read().strip()
print("DNA sequence:", dna_sequence)  # Display a sample of the sequence

motif = "CGT"
motif_positions = [i for i in range(len(dna_sequence)) if dna_sequence.startswith(motif, i)]
print("Motif", motif, "found at positions:", motif_positions)

start_codon = "ATG"
stop_codons = ["TAA", "TAG", "TGA"]
coding_regions = []

for i in range(len(dna_sequence) - 2):
    if dna_sequence[i:i + 3] == start_codon:
        for j in range(i + 3, len(dna_sequence) - 2, 3):
            if dna_sequence[j:j + 3] in stop_codons:
                coding_regions.append((i, j + 3))
                break

if coding_regions:
    print("Coding regions found:")
    for start, end in coding_regions:
        print("Start:", start, "End:", end, "Sequence:", dna_sequence[start:end])
else:
    print("No coding regions found.")

#optional
print("Summary of DNA Sequence Analysis:")
print("GC Content:", gc_content, "%")
print("Motif positions:", motif_positions)
print("Coding regions:", coding_regions)