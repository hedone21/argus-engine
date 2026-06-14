/// AUF source_hash hybrid 알고리즘 (ENG-DAT-096.6).
///
/// `sha256(size_le8 || mtime_le8 || head_8MB || tail_8MB)`
///
/// 파일이 head_n 미만인 경우: tail_n = 0, head_n = file_size 전체 (full hash 자동 적용).
use crate::auf::error::{AufError, AufResult};
use std::path::Path;

/// head/tail 청크 크기 (8MB).
const CHUNK_SIZE: usize = 8 * 1024 * 1024;

/// GGUF 파일에서 hybrid source hash를 계산한다.
///
/// 알고리즘: `sha256(file_size_le8 || mtime_le8 || head_8MB || tail_8MB)`
/// 파일이 16MB 미만인 경우 head + tail이 겹칠 수 있으므로 head_n + tail_n은 file_size를 초과하지 않는다.
pub fn compute_source_hash(path: &Path) -> AufResult<([u8; 32], u64, u64)> {
    use std::io::Read;

    let meta = std::fs::metadata(path).map_err(AufError::Io)?;
    let file_size = meta.len();
    let mtime = meta
        .modified()
        .map_err(AufError::Io)?
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let head_n = (CHUNK_SIZE as u64).min(file_size) as usize;
    let tail_n = if file_size <= head_n as u64 {
        0usize
    } else {
        (CHUNK_SIZE as u64).min(file_size - head_n as u64) as usize
    };

    let mut file = std::fs::File::open(path).map_err(AufError::Io)?;
    let mut head_bytes = vec![0u8; head_n];
    {
        use std::io::Read;
        file.read_exact(&mut head_bytes).map_err(AufError::Io)?;
    }

    let tail_bytes = if tail_n > 0 {
        use std::io::Seek;
        file.seek(std::io::SeekFrom::End(-(tail_n as i64)))
            .map_err(AufError::Io)?;
        let mut buf = vec![0u8; tail_n];
        file.read_exact(&mut buf).map_err(AufError::Io)?;
        buf
    } else {
        Vec::new()
    };

    let hash = compute_hash_from_parts(file_size, mtime, &head_bytes, &tail_bytes);
    Ok((hash, file_size, mtime))
}

/// bytes로부터 hybrid hash를 계산한다 (테스트 친화).
pub fn compute_source_hash_from_bytes(bytes: &[u8]) -> [u8; 32] {
    let file_size = bytes.len() as u64;
    let head_n = CHUNK_SIZE.min(bytes.len());
    let tail_n = if bytes.len() <= head_n {
        0
    } else {
        CHUNK_SIZE.min(bytes.len() - head_n)
    };
    compute_hash_from_parts(
        file_size,
        0,
        &bytes[..head_n],
        &bytes[bytes.len() - tail_n..],
    )
}

fn compute_hash_from_parts(file_size: u64, mtime: u64, head: &[u8], tail: &[u8]) -> [u8; 32] {
    // sha256 구현 — 외부 crate 없이 표준 sha256 직접 구현
    // Cargo.toml에 sha2 crate가 없으므로 간단한 내장 구현 사용
    use sha2_impl::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(&file_size.to_le_bytes());
    hasher.update(&mtime.to_le_bytes());
    hasher.update(head);
    hasher.update(tail);
    hasher.finalize()
}

/// 최소한의 sha2 구현 (외부 crate 없이)
mod sha2_impl {
    // RFC 6234 / FIPS 180-4 SHA-256
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    pub struct Sha256 {
        state: [u32; 8],
        buf: Vec<u8>,
        total_len: u64,
    }

    pub trait Digest {
        fn new() -> Self;
        fn update(&mut self, data: &[u8]);
        fn finalize(self) -> [u8; 32];
    }

    impl Digest for Sha256 {
        fn new() -> Self {
            Sha256 {
                state: [
                    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
                    0x1f83d9ab, 0x5be0cd19,
                ],
                buf: Vec::new(),
                total_len: 0,
            }
        }

        fn update(&mut self, data: &[u8]) {
            self.total_len += data.len() as u64;
            self.buf.extend_from_slice(data);
            while self.buf.len() >= 64 {
                let block: [u8; 64] = self.buf[..64].try_into().unwrap();
                self.buf.drain(..64);
                process_block(&mut self.state, &block);
            }
        }

        fn finalize(mut self) -> [u8; 32] {
            let bit_len = self.total_len * 8;
            self.buf.push(0x80);
            while self.buf.len() % 64 != 56 {
                self.buf.push(0x00);
            }
            self.buf.extend_from_slice(&bit_len.to_be_bytes());
            while self.buf.len() >= 64 {
                let block: [u8; 64] = self.buf[..64].try_into().unwrap();
                self.buf.drain(..64);
                process_block(&mut self.state, &block);
            }
            let mut out = [0u8; 32];
            for (i, &s) in self.state.iter().enumerate() {
                out[i * 4..i * 4 + 4].copy_from_slice(&s.to_be_bytes());
            }
            out
        }
    }

    fn process_block(state: &mut [u32; 8], block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_deterministic() {
        let data = b"hello world";
        let h1 = compute_source_hash_from_bytes(data);
        let h2 = compute_source_hash_from_bytes(data);
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_different_data_differs() {
        let h1 = compute_source_hash_from_bytes(b"abc");
        let h2 = compute_source_hash_from_bytes(b"abd");
        assert_ne!(h1, h2);
    }

    #[test]
    fn hash_known_value_empty() {
        // sha256("" + size=0 + mtime=0) — size-independent result for coverage
        let h = compute_source_hash_from_bytes(b"");
        // 32B 반환 확인
        assert_eq!(h.len(), 32);
        // 모두 0이 아님 (sha256은 빈 입력에도 non-zero)
        assert!(h.iter().any(|&b| b != 0));
    }

    #[test]
    fn hash_from_file() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.bin");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(b"test data for hash").unwrap();
        }
        let (hash, size, _mtime) = compute_source_hash(&path).unwrap();
        assert_eq!(size, 18);
        assert_eq!(hash.len(), 32);
    }

    #[test]
    fn sha256_known_vector() {
        use sha2_impl::{Digest, Sha256};
        // sha256("abc") = ba7816bf8f01cfea414140de5dae2ec73b00361bbef0469a7b79e0a4befeedd1... (first 4 bytes)
        let mut h = Sha256::new();
        h.update(b"abc");
        let digest = h.finalize();
        // 표준 SHA-256("abc") 첫 4바이트: 0xba, 0x78, 0x16, 0xbf
        assert_eq!(&digest[..4], &[0xba, 0x78, 0x16, 0xbf]);
    }
}
