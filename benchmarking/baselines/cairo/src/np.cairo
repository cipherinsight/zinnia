//! NumPy behaviours that several cases need: sorting, argsort and Python-style
//! indexing. Cairo arrays are append-only, so sorting builds new arrays (merge
//! sort).

/// `np.sort`: a stable merge sort.
pub fn sort<T, +Copy<T>, +Drop<T>, +PartialOrd<T>>(values: Span<T>) -> Array<T> {
    if values.len() <= 1 {
        let mut out = array![];
        for v in values {
            out.append(*v);
        }
        return out;
    }
    let mid = values.len() / 2;
    let left = sort(values.slice(0, mid));
    let right = sort(values.slice(mid, values.len() - mid));
    let (left, right) = (left.span(), right.span());
    let mut out = array![];
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        if *right.at(j) < *left.at(i) {
            out.append(*right.at(j));
            j += 1;
        } else {
            out.append(*left.at(i));
            i += 1;
        }
    }
    out.append_span(left.slice(i, left.len() - i));
    out.append_span(right.slice(j, right.len() - j));
    out
}

/// `np.argsort`: the indices that sort `values`, ties in index order (a
/// stable merge sort of the indices).
pub fn argsort<T, +Copy<T>, +Drop<T>, +PartialOrd<T>>(values: Span<T>) -> Array<u32> {
    let mut indices = array![];
    for i in 0..values.len() {
        indices.append(i);
    }
    argsort_indices(values, indices.span())
}

fn argsort_indices<T, +Copy<T>, +Drop<T>, +PartialOrd<T>>(
    values: Span<T>, indices: Span<u32>,
) -> Array<u32> {
    if indices.len() <= 1 {
        let mut out = array![];
        out.append_span(indices);
        return out;
    }
    let mid = indices.len() / 2;
    let left = argsort_indices(values, indices.slice(0, mid));
    let right = argsort_indices(values, indices.slice(mid, indices.len() - mid));
    let (left, right) = (left.span(), right.span());
    let mut out = array![];
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        if *values.at(*right.at(j)) < *values.at(*left.at(i)) {
            out.append(*right.at(j));
            j += 1;
        } else {
            out.append(*left.at(i));
            i += 1;
        }
    }
    out.append_span(left.slice(i, left.len() - i));
    out.append_span(right.slice(j, right.len() - j));
    out
}

/// A Python/NumPy index into an axis of length `len`: negative indices count
/// from the end; anything outside `[-len, len)` fails like NumPy's IndexError.
pub fn np_index(i: i64, len: u32) -> u32 {
    let n: i64 = len.into();
    assert!(-n <= i && i < n, "index out of range");
    let wrapped = if i < 0 {
        i + n
    } else {
        i
    };
    wrapped.try_into().unwrap()
}

#[cfg(test)]
mod tests {
    use super::{argsort, np_index, sort};

    #[test]
    fn sorts() {
        assert!(sort(array![3_i64, -1, 2, -1, 0].span()) == array![-1, -1, 0, 2, 3]);
        assert!(argsort(array![3_i64, -1, 2, -1, 0].span()) == array![1, 3, 4, 2, 0]);
    }

    #[test]
    fn indexes_like_numpy() {
        assert!(np_index(-1, 5) == 4);
        assert!(np_index(2, 5) == 2);
    }
}
