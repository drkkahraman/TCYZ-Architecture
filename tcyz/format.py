import struct
import numpy as np
import torch

TYPE_INT32 = 0
TYPE_FLOAT32 = 1
TYPE_STRING = 2
TYPE_ARRAY_INT32 = 3
TYPE_ARRAY_FLOAT32 = 4

DTYPE_MAP = {
    0: torch.float32,
    1: torch.float16,
    2: torch.bfloat16,
    3: torch.int8
}

INV_DTYPE_MAP = {v: k for k, v in DTYPE_MAP.items()}

NP_DTYPE_MAP = {
    0: np.float32,
    1: np.float16,
    2: None,
    3: np.int8
}

def pack_string(s):
    encoded = s.encode('utf-8')
    return struct.pack('<I', len(encoded)) + encoded

def unpack_string(f):
    length_data = f.read(4)
    if not length_data:
        return None
    length = struct.unpack('<I', length_data)[0]
    return f.read(length).decode('utf-8')

def write_tcyz(file_path, metadata, tensors, alignment=64):
    with open(file_path, 'wb') as f:
        f.write(b'TCYZ')
        f.write(struct.pack('<I', 1))
        f.write(struct.pack('<Q', len(metadata)))
        f.write(struct.pack('<Q', len(tensors)))

        for key, (val_type, val) in metadata.items():
            f.write(pack_string(key))
            f.write(struct.pack('<I', val_type))
            if val_type == TYPE_INT32:
                f.write(struct.pack('<i', val))
            elif val_type == TYPE_FLOAT32:
                f.write(struct.pack('<f', val))
            elif val_type == TYPE_STRING:
                f.write(pack_string(val))
            elif val_type == TYPE_ARRAY_INT32:
                f.write(struct.pack('<I', len(val)))
                f.write(struct.pack(f'<{len(val)}i', *val))
            elif val_type == TYPE_ARRAY_FLOAT32:
                f.write(struct.pack('<I', len(val)))
                f.write(struct.pack(f'<{len(val)}f', *val))

        tensor_info_list = []
        offset = 0
        for name, tensor in tensors.items():
            dtype_id = INV_DTYPE_MAP[tensor.dtype]
            shape = list(tensor.shape)
            
            element_size = tensor.element_size()
            size_bytes = tensor.numel() * element_size
            
            pad = (alignment - (size_bytes % alignment)) % alignment
            padded_size = size_bytes + pad
            
            tensor_info_list.append({
                'name': name,
                'shape': shape,
                'dtype_id': dtype_id,
                'offset': offset,
                'size_bytes': size_bytes,
                'padded_size': padded_size,
                'tensor': tensor
            })
            offset += padded_size

        for info in tensor_info_list:
            f.write(pack_string(info['name']))
            f.write(struct.pack('<I', len(info['shape'])))
            f.write(struct.pack(f"<{len(info['shape'])}I", *info['shape']))
            f.write(struct.pack('<I', info['dtype_id']))
            f.write(struct.pack('<Q', info['offset']))
            f.write(struct.pack('<Q', info['size_bytes']))

        curr_pos = f.tell()
        padding_needed = (alignment - (curr_pos % alignment)) % alignment
        if padding_needed > 0:
            f.write(b'\x00' * padding_needed)

        data_start = f.tell()
        for info in tensor_info_list:
            t = info['tensor'].contiguous()
            if t.device.type != 'cpu':
                t = t.cpu()
            
            storage = t.untyped_storage()
            f.write(bytes(storage))
            
            pad = info['padded_size'] - info['size_bytes']
            if pad > 0:
                f.write(b'\x00' * pad)

def read_tcyz(file_path):
    f = open(file_path, 'rb')
    magic = f.read(4)
    if magic != b'TCYZ':
        raise ValueError("Invalid magic bytes")
    
    version = struct.unpack('<I', f.read(4))[0]
    if version != 1:
        raise ValueError(f"Unsupported version: {version}")
        
    num_metadata = struct.unpack('<Q', f.read(8))[0]
    num_tensors = struct.unpack('<Q', f.read(8))[0]
    
    metadata = {}
    for _ in range(num_metadata):
        key = unpack_string(f)
        val_type = struct.unpack('<I', f.read(4))[0]
        if val_type == TYPE_INT32:
            val = struct.unpack('<i', f.read(4))[0]
        elif val_type == TYPE_FLOAT32:
            val = struct.unpack('<f', f.read(4))[0]
        elif val_type == TYPE_STRING:
            val = unpack_string(f)
        elif val_type == TYPE_ARRAY_INT32:
            length = struct.unpack('<I', f.read(4))[0]
            val = list(struct.unpack(f'<{length}i', f.read(length * 4)))
        elif val_type == TYPE_ARRAY_FLOAT32:
            length = struct.unpack('<I', f.read(4))[0]
            val = list(struct.unpack(f'<{length}f', f.read(length * 4)))
        else:
            raise ValueError(f"Unknown metadata value type: {val_type}")
        metadata[key] = val
        
    tensors_info = []
    for _ in range(num_tensors):
        name = unpack_string(f)
        num_dims = struct.unpack('<I', f.read(4))[0]
        shape = list(struct.unpack(f'<{num_dims}I', f.read(num_dims * 4)))
        dtype_id = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        size_bytes = struct.unpack('<Q', f.read(8))[0]
        
        tensors_info.append({
            'name': name,
            'shape': shape,
            'dtype_id': dtype_id,
            'offset': offset,
            'size_bytes': size_bytes
        })
        
    curr_pos = f.tell()
    alignment = 64
    padding_needed = (alignment - (curr_pos % alignment)) % alignment
    data_start = curr_pos + padding_needed
    f.close()
    
    loaded_tensors = {}
    
    for info in tensors_info:
        dtype = DTYPE_MAP[info['dtype_id']]
        shape = info['shape']
        offset = data_start + info['offset']
        size_bytes = info['size_bytes']
        
        if dtype == torch.bfloat16:
            mmap_arr = np.memmap(file_path, dtype=np.uint16, mode='r', offset=offset, shape=(size_bytes // 2,))
            tensor_cpu = torch.from_numpy(mmap_arr).view(shape).view(torch.bfloat16)
        else:
            np_dtype = NP_DTYPE_MAP[info['dtype_id']]
            mmap_arr = np.memmap(file_path, dtype=np_dtype, mode='r', offset=offset, shape=tuple(shape))
            tensor_cpu = torch.from_numpy(mmap_arr)
            
        loaded_tensors[info['name']] = tensor_cpu
        
    return metadata, loaded_tensors
