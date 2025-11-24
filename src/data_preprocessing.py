import numpy as np
from scapy.all import rdpcap, IP, TCP, UDP
from typing import List, Tuple, Optional
import os

class TrafficPreprocessor:
    def __init__(self, max_packets=196, patch_size=16):
        self.max_packets = max_packets
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size
        self.network_header_bytes = 20
        self.transport_header_bytes = 20
        self.payload_bytes = 216
        self.image_size = 224
        assert self.network_header_bytes + self.transport_header_bytes + self.payload_bytes == self.patch_dim, \
            f"Patch structure must equal patch_dim: {self.network_header_bytes}+{self.transport_header_bytes}+{self.payload_bytes} != {self.patch_dim}"
        
    def pcap2flow(self, pcap_file: str) -> dict:
        packets = rdpcap(pcap_file)
        flows = {}
        
        for packet in packets:
            if IP in packet:
                if TCP in packet or UDP in packet:
                    src_ip = packet[IP].src
                    dst_ip = packet[IP].dst
                    
                    if TCP in packet:
                        src_port = packet[TCP].sport
                        dst_port = packet[TCP].dport
                        protocol = 'TCP'
                    else:
                        src_port = packet[UDP].sport
                        dst_port = packet[UDP].dport
                        protocol = 'UDP'
                    
                    flow_key = (src_ip, dst_ip, src_port, dst_port, protocol)
                    
                    if flow_key not in flows:
                        flows[flow_key] = []
                    flows[flow_key].append(packet)
        
        return flows
    
    def packet2patch(self, packet) -> np.ndarray:
        patch = np.zeros(self.patch_dim, dtype=np.uint8)
        
        if IP not in packet:
            return patch
        
        try:
            ip_layer = packet[IP]
            ip_bytes = bytes(ip_layer)[:self.network_header_bytes]
            patch[:len(ip_bytes)] = list(ip_bytes)
            offset = self.network_header_bytes
            
            if TCP in packet:
                tcp_layer = packet[TCP]
                tcp_bytes = bytes(tcp_layer)[:self.transport_header_bytes]
                patch[offset:offset+len(tcp_bytes)] = list(tcp_bytes)
                offset += self.transport_header_bytes
                
                if hasattr(tcp_layer, 'payload') and tcp_layer.payload:
                    payload_bytes = bytes(tcp_layer.payload)[:self.payload_bytes]
                    patch[offset:offset+len(payload_bytes)] = list(payload_bytes)
                    
            elif UDP in packet:
                udp_layer = packet[UDP]
                udp_bytes = bytes(udp_layer)[:self.transport_header_bytes]
                patch[offset:offset+len(udp_bytes)] = list(udp_bytes)
                offset += self.transport_header_bytes
                
                if hasattr(udp_layer, 'payload') and udp_layer.payload:
                    payload_bytes = bytes(udp_layer.payload)[:self.payload_bytes]
                    patch[offset:offset+len(payload_bytes)] = list(payload_bytes)
        except Exception as e:
            pass
        
        return patch
    
    def flow2image(self, flow_packets: List) -> np.ndarray:
        patches = []
        handshake_idx = self._find_handshake_packet(flow_packets)
        
        for i, packet in enumerate(flow_packets):
            if i >= self.max_packets:
                break
            patch = self.packet2patch(packet)
            patches.append(patch)
        
        if len(patches) < self.max_packets:
            if len(patches) > 0:
                if handshake_idx is not None and handshake_idx < len(patches):
                    padding_patch = patches[handshake_idx]
                else:
                    padding_patch = patches[-1]
            else:
                padding_patch = np.zeros(self.patch_dim, dtype=np.uint8)
            
            while len(patches) < self.max_packets:
                patches.append(padding_patch.copy())
        
        patches_array = np.array(patches, dtype=np.uint8)
        
        grid_size = int(np.sqrt(self.max_packets))
        patches_reshaped = patches_array.reshape(grid_size, grid_size, self.patch_size, self.patch_size)
        
        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        for i in range(grid_size):
            for j in range(grid_size):
                row_start = i * self.patch_size
                col_start = j * self.patch_size
                image[row_start:row_start+self.patch_size, col_start:col_start+self.patch_size] = patches_reshaped[i, j]
        
        image_r = image.copy()
        image_g = np.roll(image, 1, axis=0)
        image_b = np.roll(image, 2, axis=0)
        
        image_rgb = np.stack([image_r, image_g, image_b], axis=0)
        
        return image_rgb
    
    def _find_handshake_packet(self, flow_packets: List) -> Optional[int]:
        for i, packet in enumerate(flow_packets):
            if TCP in packet:
                tcp_layer = packet[TCP]
                if tcp_layer.flags & 0x02:
                    return i
        return None
    
    def preprocess_pcap(self, pcap_file: str) -> List[np.ndarray]:
        flows = self.pcap2flow(pcap_file)
        images = []
        
        for flow_key, flow_packets in flows.items():
            image = self.flow2image(flow_packets)
            images.append(image)
        
        return images
    
    def preprocess_dataset(self, pcap_dir: str, labels_dict: dict) -> Tuple[np.ndarray, np.ndarray]:
        all_images = []
        all_labels = []
        
        for filename in os.listdir(pcap_dir):
            if filename.endswith('.pcap'):
                pcap_path = os.path.join(pcap_dir, filename)
                images = self.preprocess_pcap(pcap_path)
                
                label = labels_dict.get(filename, 0)
                
                all_images.extend(images)
                all_labels.extend([label] * len(images))
        
        return np.array(all_images), np.array(all_labels)
