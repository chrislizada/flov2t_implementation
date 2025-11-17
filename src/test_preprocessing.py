import unittest
import numpy as np
from data_preprocessing import TrafficPreprocessor
from scapy.all import IP, TCP, UDP, Ether
from scapy.layers.inet import Raw


class TestTrafficPreprocessor(unittest.TestCase):
    
    def setUp(self):
        self.preprocessor = TrafficPreprocessor(max_packets=196, patch_size=16)
    
    def test_patch_dimensions(self):
        self.assertEqual(self.preprocessor.patch_dim, 256)
        self.assertEqual(
            self.preprocessor.network_header_bytes + 
            self.preprocessor.transport_header_bytes + 
            self.preprocessor.payload_bytes,
            256
        )
    
    def test_packet2patch_structure(self):
        packet = Ether() / IP(src="192.168.1.1", dst="192.168.1.2") / TCP(sport=12345, dport=80) / Raw(load=b"A" * 100)
        
        patch = self.preprocessor.packet2patch(packet)
        
        self.assertEqual(patch.shape, (256,))
        self.assertEqual(patch.dtype, np.uint8)
        
        self.assertTrue(np.any(patch[:20] != 0))
        self.assertTrue(np.any(patch[20:40] != 0))
    
    def test_packet2patch_no_ip(self):
        packet = Ether()
        patch = self.preprocessor.packet2patch(packet)
        
        self.assertEqual(patch.shape, (256,))
        self.assertTrue(np.all(patch == 0))
    
    def test_flow2image_shape(self):
        packets = []
        for i in range(10):
            packet = Ether() / IP(src="192.168.1.1", dst="192.168.1.2") / TCP(sport=12345, dport=80) / Raw(load=b"Test")
            packets.append(packet)
        
        image = self.preprocessor.flow2image(packets)
        
        self.assertEqual(image.shape, (3, 224, 224))
        self.assertEqual(image.dtype, np.uint8)
    
    def test_flow2image_padding(self):
        packets = []
        for i in range(5):
            packet = Ether() / IP(src="192.168.1.1", dst="192.168.1.2") / TCP(sport=12345, dport=80)
            packets.append(packet)
        
        image = self.preprocessor.flow2image(packets)
        
        self.assertEqual(image.shape, (3, 224, 224))
    
    def test_flow2image_max_packets(self):
        packets = []
        for i in range(300):
            packet = Ether() / IP(src="192.168.1.1", dst="192.168.1.2") / TCP(sport=12345, dport=80)
            packets.append(packet)
        
        image = self.preprocessor.flow2image(packets)
        
        self.assertEqual(image.shape, (3, 224, 224))
    
    def test_handshake_detection(self):
        packets = []
        
        syn_packet = Ether() / IP(src="192.168.1.1", dst="192.168.1.2") / TCP(sport=12345, dport=80, flags="S")
        packets.append(syn_packet)
        
        for i in range(5):
            packet = Ether() / IP(src="192.168.1.1", dst="192.168.1.2") / TCP(sport=12345, dport=80, flags="A")
            packets.append(packet)
        
        handshake_idx = self.preprocessor._find_handshake_packet(packets)
        
        self.assertEqual(handshake_idx, 0)
    
    def test_image_rgb_channels(self):
        packets = []
        for i in range(10):
            packet = Ether() / IP(src="192.168.1.1", dst="192.168.1.2") / TCP(sport=12345, dport=80)
            packets.append(packet)
        
        image = self.preprocessor.flow2image(packets)
        
        self.assertEqual(image.shape[0], 3)
        
        np.testing.assert_array_equal(image[0], image[1])
        np.testing.assert_array_equal(image[1], image[2])


class TestPcap2Flow(unittest.TestCase):
    
    def setUp(self):
        self.preprocessor = TrafficPreprocessor()
    
    def test_pcap2flow_basic(self):
        import tempfile
        from scapy.all import wrpcap
        
        packets = []
        for i in range(5):
            packet = Ether() / IP(src="192.168.1.1", dst="192.168.1.2") / TCP(sport=12345, dport=80)
            packets.append(packet)
        
        for i in range(3):
            packet = Ether() / IP(src="192.168.1.3", dst="192.168.1.4") / UDP(sport=5353, dport=5353)
            packets.append(packet)
        
        with tempfile.NamedTemporaryFile(suffix='.pcap', delete=False) as f:
            pcap_path = f.name
            wrpcap(pcap_path, packets)
        
        flows = self.preprocessor.pcap2flow(pcap_path)
        
        self.assertGreater(len(flows), 0)
        
        import os
        os.unlink(pcap_path)


if __name__ == '__main__':
    unittest.main()
