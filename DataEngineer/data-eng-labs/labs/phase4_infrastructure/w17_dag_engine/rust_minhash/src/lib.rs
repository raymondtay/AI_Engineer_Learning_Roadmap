//! Week 17 hot path: MinHash signatures over word 5-gram shingles, exposed with a
//! C ABI so Python can call it through ctypes (no pyo3, no network needed to build).
//!
//! Must match `minhash_py` in solution.py bit for bit:
//!   shingle hash  h = FNV-1a 64 over the UTF-8 bytes of "w1 w2 w3 w4 w5"
//!   permutation i (a_i, b_i) from splitmix64(seed + i), a_i forced odd
//!   sig[i] = min over shingles of (a_i * h + b_i) mod 2^64  (wrapping)
//!   a doc with fewer than 5 words is one shingle: the whole doc

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn fnv1a(parts: &[&[u8]]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for (j, p) in parts.iter().enumerate() {
        if j > 0 {
            h ^= b' ' as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        for &c in p.iter() {
            h ^= c as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// Signatures for `n_docs` docs packed in `buf`; doc d is buf[offsets[d]..offsets[d+1]].
/// Writes n_docs * k u64s into `out`.
#[no_mangle]
pub extern "C" fn minhash_batch(buf: *const u8, offsets: *const u64, n_docs: u64, k: u64, seed: u64, out: *mut u64) {
    let k = k as usize;
    let offs = unsafe { std::slice::from_raw_parts(offsets, n_docs as usize + 1) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, n_docs as usize * k) };
    let mut a = vec![0u64; k];
    let mut b = vec![0u64; k];
    for i in 0..k {
        a[i] = splitmix64(seed.wrapping_add(2 * i as u64)) | 1;
        b[i] = splitmix64(seed.wrapping_add(2 * i as u64 + 1));
    }
    let mut words: Vec<&[u8]> = Vec::new();
    for d in 0..n_docs as usize {
        let doc = unsafe { std::slice::from_raw_parts(buf.add(offs[d] as usize), (offs[d + 1] - offs[d]) as usize) };
        words.clear();
        words.extend(doc.split(|&c| c == b' ' || c == b'\n' || c == b'\t').filter(|w| !w.is_empty()));
        let sig = &mut out[d * k..(d + 1) * k];
        sig.iter_mut().for_each(|s| *s = u64::MAX);
        let n_sh = if words.len() < 5 { 1 } else { words.len() - 4 };
        for s in 0..n_sh {
            let end = (s + 5).min(words.len());
            let h = fnv1a(&words[s..end]);
            for i in 0..k {
                let v = a[i].wrapping_mul(h).wrapping_add(b[i]);
                if v < sig[i] {
                    sig[i] = v;
                }
            }
        }
    }
}
