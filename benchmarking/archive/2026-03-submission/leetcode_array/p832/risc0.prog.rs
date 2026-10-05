use risc0_zkvm::guest::env;

fn main() {
    // TODO: Implement your guest code here
    // read the input
    let mut image: Vec<Vec<u32>> = Vec::new();
    let mut result: Vec<Vec<u32>> = Vec::new();

    for i in 0..10 {
        let mut tmp = Vec::new();
        for j in 0..10 {
            let n = env::read();
            tmp.push(n);
        }
        image.push(tmp);
    }
    for i in 0..10 {
        let mut tmp = Vec::new();
        for j in 0..10 {
            let n = env::read();
            tmp.push(n);
        }
        result.push(tmp);
    }

    // Check that both image and result are 10x10 arrays.
    assert_eq!(image.len(), 10, "Image must have 10 rows");
    assert_eq!(result.len(), 10, "Result must have 10 rows");
    for (i, row) in image.iter().enumerate() {
        assert_eq!(row.len(), 10, "Row {} in image must have 10 columns", i);
        for (j, &val) in row.iter().enumerate() {
            assert!(
                val == 0 || val == 1,
                "Image element at ({},{}) must be 0 or 1",
                i,
                j
            );
        }
    }
    for (i, row) in result.iter().enumerate() {
        assert_eq!(row.len(), 10, "Row {} in result must have 10 columns", i);
        for (j, &val) in row.iter().enumerate() {
            assert!(
                val == 0 || val == 1,
                "Result element at ({},{}) must be 0 or 1",
                i,
                j
            );
        }
    }

    // For each element, verify that result[i][j] == 1 - image[i][9 - j]
    for i in 0..10 {
        for j in 0..10 {
            assert_eq!(
                result[i][j],
                1 - image[i][10 - 1 - j],
                "Mismatch at position ({},{}): expected {} but got {}",
                i,
                j,
                1 - image[i][10 - 1 - j],
                result[i][j]
            );
        }
    }

    // write public output to the journal
    // env::commit(&input);
}
