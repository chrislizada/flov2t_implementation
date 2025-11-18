# Using CICIDS2017/2018 PCAP Data with FLoV2T

This guide explains how to train FLoV2T with real CICIDS2017 or CICIDS2018 PCAP files.

## Quick Start

```bash
# Train with CICIDS2017 PCAP files
python src/main.py --data_path /path/to/cicids2017/pcaps --dataset CICIDS2017

# Train with CICIDS2018 PCAP files
python src/main.py --data_path /path/to/cicids2018/pcaps --dataset CICIDS2018
```

## PCAP File Naming Convention

The data loader automatically detects attack types from PCAP filenames using keywords.

### CICIDS2017 Keywords
Your PCAP filenames should contain these keywords:

| Attack Type | Keywords | Label ID |
|-------------|----------|----------|
| Botnet | `botnet`, `bot` | 0 |
| DoS-Slowloris | `slowloris`, `slow` | 1 |
| DoS-GoldenEye | `goldeneye`, `golden` | 2 |
| DoS-Hulk | `hulk` | 3 |
| SSH-BruteForce | `ssh`, `patator`, `ftp` | 4 |
| Web-SQL | `sql` | 5 |
| Web-XSS | `xss` | 6 |
| Web-Bruteforce | `bruteforce`, `brute` | 7 |

**Example filenames:**
- `botnet_traffic.pcap` → Label 0 (Botnet)
- `dos_slowloris_attack.pcap` → Label 1 (DoS-Slowloris)
- `ssh_bruteforce.pcap` → Label 4 (SSH-BruteForce)
- `web_sql_injection.pcap` → Label 5 (Web-SQL)

### CICIDS2018 Keywords
Similar structure for CICIDS2018:

| Attack Type | Keywords | Label ID |
|-------------|----------|----------|
| Bot | `bot` | 0 |
| DoS-SlowHTTPTest | `slowhttptest`, `slow` | 1 |
| DoS-GoldenEye | `goldeneye`, `golden` | 2 |
| DoS-Hulk/DDoS | `hulk`, `ddos` | 3 |
| SSH-Bruteforce | `ssh`, `bruteforce` | 4 |
| SQL-Injection | `sql`, `injection` | 5 |
| XSS | `xss` | 6 |
| Brute-Force | `brute` | 7 |

**Files containing `benign` or `normal` are automatically skipped.**

## Directory Structure

```
/path/to/cicids2017/pcaps/
├── botnet_capture.pcap
├── dos_slowloris.pcap
├── dos_goldeneye.pcap
├── dos_hulk.pcap
├── ssh_patator.pcap
├── web_sql_injection.pcap
├── web_xss_attack.pcap
└── web_bruteforce.pcap
```

Or with subdirectories (automatically searched recursively):
```
/path/to/cicids2017/
├── monday/
│   ├── botnet.pcap
│   └── dos_slowloris.pcap
├── tuesday/
│   ├── dos_goldeneye.pcap
│   └── ssh_attack.pcap
└── wednesday/
    └── web_attacks.pcap
```

## Training Commands

### Basic Training
```bash
python src/main.py \
  --data_path /path/to/pcaps \
  --dataset CICIDS2017 \
  --num_clients 3 \
  --scenario iid
```

### Non-IID Scenario
```bash
python src/main.py \
  --data_path /path/to/pcaps \
  --dataset CICIDS2017 \
  --num_clients 3 \
  --scenario non-iid \
  --aggregation rgpa
```

### Full Configuration
```bash
python src/main.py \
  --data_path /path/to/pcaps \
  --dataset CICIDS2017 \
  --num_clients 5 \
  --scenario non-iid \
  --num_rounds 20 \
  --local_epochs 5 \
  --batch_size 32 \
  --lr 0.001 \
  --aggregation rgpa \
  --checkpoint_dir ./checkpoints \
  --seed 42
```

### Testing with Limited Data
For quick testing with limited samples per PCAP file:
```bash
python src/main.py \
  --data_path /path/to/pcaps \
  --dataset CICIDS2017 \
  --max_samples_per_file 100
```

## Data Processing Pipeline

The system automatically:

1. **Finds PCAP files** - Recursively searches for `.pcap`, `.pcapng`, `.cap` files
2. **Extracts labels** - From filenames using keyword matching
3. **Processes flows** - Uses Pcap2Flow to divide traffic into bidirectional flows
4. **Creates patches** - Converts packets to 16×16 patches (256 bytes each)
5. **Generates images** - Creates 224×224 RGB images from 196 patches (14×14 grid)
6. **Splits data** - 80% training, 20% testing
7. **Partitions clients** - IID or non-IID distribution based on scenario

## Expected Output

```
================================================================================
Loading CICIDS2017 dataset from PCAP files
================================================================================
Directory: /path/to/pcaps

Found 8 PCAP files

→ Processing: botnet_capture.pcap
  Label: 0 (Botnet)
  Flows extracted: 246

→ Processing: dos_slowloris.pcap
  Label: 1 (DoS-Slowloris)
  Flows extracted: 684

⊘ Skipping: benign_traffic.pcap (benign/unknown)

================================================================================
Dataset loaded successfully
================================================================================
Total samples: 9118
Data shape: (9118, 3, 224, 224)
================================================================================

Class Distribution:
--------------------------------------------------------------------------------
  Class 0 - Botnet              :    246 samples ( 2.70%)
  Class 1 - DoS-Slowloris       :    684 samples ( 7.50%)
  Class 2 - DoS-Goldeneye       :   2494 samples (27.35%)
  Class 3 - DoS-Hulk            :   4660 samples (51.12%)
  Class 4 - SSH-BruteForce      :    993 samples (10.89%)
  Class 5 - Web-SQL             :      5 samples ( 0.05%)
  Class 6 - Web-XSS             :      6 samples ( 0.07%)
  Class 7 - Web-Bruteforce      :     30 samples ( 0.33%)
--------------------------------------------------------------------------------
```

## Troubleshooting

### No PCAP files found
```
ValueError: No PCAP files found in /path/to/pcaps
```
**Solution**: Check the path and ensure PCAP files have `.pcap`, `.pcapng`, or `.cap` extensions.

### All files skipped
```
ValueError: No data was successfully loaded. Check PCAP filenames contain attack keywords.
```
**Solution**: Rename PCAP files to include attack keywords (e.g., `botnet`, `dos`, `sql`).

### Memory errors with large PCAPs
```
MemoryError: Unable to allocate array
```
**Solution**: Use `--max_samples_per_file` to limit flows per file:
```bash
python src/main.py --data_path /path/to/pcaps --max_samples_per_file 500
```

### Scapy errors
```
Error processing file.pcap: [Scapy error]
```
**Solution**: PCAP might be corrupted. Skip it or repair with:
```bash
tcpdump -r corrupted.pcap -w fixed.pcap
```

## Performance Tips

### For large datasets (>10GB)
1. Process files in batches using `--max_samples_per_file 1000`
2. Use more powerful instance (e.g., g4dn.xlarge with 16GB RAM)
3. Reduce batch size: `--batch_size 16`

### For quick testing
1. Use small subset: `--max_samples_per_file 100`
2. Reduce rounds: `--num_rounds 5`
3. Use CPU mode on small data

### For reproducing paper results
1. Use full CICIDS2017/2018 datasets
2. Follow paper configuration: `--num_clients 3 --scenario non-iid --aggregation rgpa`
3. Run 20 rounds: `--num_rounds 20`

## Validation

After loading data, check:
- ✓ Data shape is `(N, 3, 224, 224)` where N = number of flows
- ✓ Labels are in range [0, 7]
- ✓ Class distribution matches your PCAP files
- ✓ All 8 attack classes are represented (for full reproduction)

## Next Steps

After successful data loading:
1. Training starts automatically
2. Monitor accuracy and F1 scores each round
3. Checkpoints saved to `./checkpoints/`
4. Best model saved when accuracy improves
5. Final results compared to paper baselines

Target performance (from paper):
- **IID**: Accuracy ~97.26%, F1 ~96.99%
- **Non-IID**: Accuracy ~96.17%, F1 ~95.81%
