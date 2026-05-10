import sys

def parse_kfd_topology(filepath):
    # Initialize 8x8 matrices with 0s
    links_matrix = [[0 for _ in range(8)] for _ in range(8)]
    bw_matrix = [[0 for _ in range(8)] for _ in range(8)]
    
    current_link = {}

    def process_link(link):
        if link.get("type") == 11:
            node_from = link.get("node_from", -1)
            node_to = link.get("node_to", -1)
            
            # In AMD's driver, GPUs map to KFD nodes 4-11
            if 4 <= node_from <= 11 and 4 <= node_to <= 11:
                gcd_from = node_from - 4
                gcd_to = node_to - 4
                max_bw = link.get("max_bandwidth", 0)
                
                # Calculate links and GB/s
                links = max_bw // 50000
                bw_gb = max_bw // 1000  # Convert MB/s to GB/s
                
                links_matrix[gcd_from][gcd_to] = links
                bw_matrix[gcd_from][gcd_to] = bw_gb

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: 
                    continue
                
                parts = line.split()
                key = parts[0]
                value = parts[1] if len(parts) > 1 else ""

                if key == "type":
                    process_link(current_link)
                    current_link = {"type": int(value)}
                else:
                    try:
                        current_link[key] = int(value)
                    except ValueError:
                        current_link[key] = value
            # Catch the final block
            process_link(current_link)
            
    except FileNotFoundError:
        print(f"Error: Could not find file '{filepath}'")
        sys.exit(1)

    # Helper function to print a clean 8x8 matrix
    def print_matrix(title, matrix):
        print(f"======================= {title:^30} =======================")
        
        # Print Header
        header = f"{'':<7}"
        for i in range(8):
            header += f"GPU{i:<10}"
        print(header)
        
        # Print Rows
        for i in range(8):
            row_str = f"GPU{i:<4}"
            for j in range(8):
                if i == j:
                    val = "-"  # Mark self-connections with a dash
                else:
                    val = str(matrix[i][j])
                row_str += f"{val:<13}"
            print(row_str)
        print("==================================================================================\n")

    # Output the matrices
    print_matrix("Number of Parallel xGMI Links", links_matrix)
    print_matrix("Max Bandwidth (GB/s)", bw_matrix)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_kfd_matrix.py <topology_dump.txt>")
        sys.exit(1)
    
    parse_kfd_topology(sys.argv[1])