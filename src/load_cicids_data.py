import os
import numpy as np
from typing import Tuple, Optional, List
from data_preprocessing import TrafficPreprocessor
from config import FLoV2TConfig
import glob

class CICIDSDataLoader:
    def __init__(self, preprocessor: TrafficPreprocessor):
        self.preprocessor = preprocessor
        self.config = FLoV2TConfig()
        
        self.cicids2017_keywords = {
            'botnet': 0, 'bot': 0,
            'slowloris': 1, 'slow': 1,
            'goldeneye': 2, 'golden': 2,
            'hulk': 3,
            'ssh': 4, 'patator': 4, 'ftp': 4,
            'sql': 5,
            'xss': 6,
            'bruteforce': 7, 'brute': 7
        }
        
        self.cicids2018_keywords = {
            'bot': 0,
            'slowhttptest': 1, 'slow': 1,
            'goldeneye': 2, 'golden': 2,
            'hulk': 3, 'ddos': 3,
            'ssh': 4, 'bruteforce': 4,
            'sql': 5, 'injection': 5,
            'xss': 6,
            'brute': 7
        }
    
    def load_pcap_directory(self, data_path: str, dataset: str = 'CICIDS2017', 
                           max_samples_per_file: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        print(f"\n{'='*80}")
        print(f"Loading {dataset} dataset from PCAP files")
        print(f"{'='*80}")
        print(f"Directory: {data_path}\n")
        
        pcap_files = self._find_pcap_files(data_path)
        if not pcap_files:
            raise ValueError(f"No PCAP files found in {data_path}")
        
        print(f"Found {len(pcap_files)} PCAP files\n")
        
        keywords = self.cicids2017_keywords if dataset == 'CICIDS2017' else self.cicids2018_keywords
        
        all_images = []
        all_labels = []
        
        for pcap_file in pcap_files:
            label = self._extract_label_from_filename(pcap_file, keywords)
            
            filename = os.path.basename(pcap_file)
            
            if label is None:
                print(f"[SKIP] {filename} (benign/unknown)")
                continue
            
            class_name = self.config.ATTACK_CATEGORIES[label]
            print(f"[PROC] {filename}")
            print(f"  Label: {label} ({class_name})")
            
            try:
                images = self.preprocessor.preprocess_pcap(pcap_file)
                
                if max_samples_per_file and len(images) > max_samples_per_file:
                    images = images[:max_samples_per_file]
                
                all_images.extend(images)
                all_labels.extend([label] * len(images))
                
                print(f"  Flows extracted: {len(images)}\n")
            except Exception as e:
                print(f"  [ERROR] {e}\n")
                continue
        
        if len(all_images) == 0:
            raise ValueError("No data was successfully loaded. Check PCAP filenames contain attack keywords.")
        
        data = np.array(all_images, dtype=np.uint8)
        labels = np.array(all_labels, dtype=np.int64)
        
        print(f"{'='*80}")
        print(f"Dataset loaded successfully")
        print(f"{'='*80}")
        print(f"Total samples: {len(data)}")
        print(f"Data shape: {data.shape}")
        print(f"{'='*80}\n")
        
        self._print_class_distribution(labels)
        
        return data, labels
    
    def _find_pcap_files(self, directory: str) -> List[str]:
        pcap_patterns = ['*.pcap', '*.pcapng', '*.cap']
        pcap_files = []
        
        for pattern in pcap_patterns:
            pcap_files.extend(glob.glob(os.path.join(directory, pattern)))
            pcap_files.extend(glob.glob(os.path.join(directory, '**', pattern), recursive=True))
        
        return sorted(list(set(pcap_files)))
    
    def _extract_label_from_filename(self, filename: str, keywords: dict) -> Optional[int]:
        basename = os.path.basename(filename).lower().replace('-', '').replace('_', '')
        
        for keyword, label_id in keywords.items():
            if keyword in basename:
                return label_id
        
        if 'benign' in basename or 'normal' in basename:
            return None
        
        return None
    
    def _print_class_distribution(self, labels: np.ndarray):
        print("Class Distribution:")
        print("-" * 80)
        unique, counts = np.unique(labels, return_counts=True)
        for label_id, count in zip(unique, counts):
            class_name = self.config.ATTACK_CATEGORIES.get(label_id, 'Unknown')
            percentage = (count / len(labels)) * 100
            print(f"  Class {label_id} - {class_name:20s}: {count:6d} samples ({percentage:5.2f}%)")
        print("-" * 80 + "\n")

def load_cicids_data(data_path: str, dataset: str = 'CICIDS2017', 
                     max_samples_per_file: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    preprocessor = TrafficPreprocessor()
    loader = CICIDSDataLoader(preprocessor)
    return loader.load_pcap_directory(data_path, dataset, max_samples_per_file)
