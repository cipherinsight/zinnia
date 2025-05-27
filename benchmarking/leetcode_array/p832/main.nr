fn main(image: [[Field; 10]; 10], result: [[Field; 10]; 10]) {
    for i in 0..10 {
        for j in 0..10 {
            let p = image[i][j];
            assert(
                (p == 0) | (p == 1),
                "Error"
            );
            let r = result[i][j];
            assert((r == 0) | (r == 1), "Error");
        }
    }

    for i in 0..10 {
        for j in 0..10 {
            assert(result[i][j] == 1 - image[i][9 - j], "Error");
        }
    }
}
