#[executable]
pub fn main() {
    let mut w: u32 = 999;
    let area: u32 = 999;
    let expected_l: u32 = 37;
    let expected_w: u32 = 27;
    let mut done: bool = false;

    for i in 1_u32..1001_u32 {
        let fi = i;
        if !done {
            let q: u32 = area / fi;
            let r: u32 = area % fi;
            if r == 0 {
                w = fi;
            }
            let isq = i * i;
            let breakcond: bool = isq >= area;
            if breakcond {
                done = true;
            }
        }
    }

    let answer_w: u32 = w;
    let answer_l: u32 = area / w;

    if answer_l < answer_w {
        let tmp = answer_l;
        let answer_l = answer_w;
        let answer_w = tmp;
        assert!(answer_l == expected_l);
        assert!(answer_w == expected_w);
    } else {
        assert!(answer_l == expected_l);
        assert!(answer_w == expected_w);
    }
}
