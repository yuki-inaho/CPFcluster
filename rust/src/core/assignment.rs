//! Assignment step (Algorithm 2, Steps 17-23).

use crate::types::NO_PARENT;

pub fn assign_labels_for_component(
    cc_idx: &[usize],
    big_brother: &[i32],
    centers: &[usize],
    label_offset: i32,
) -> (i32, Vec<i32>) {
    let nc = cc_idx.len();
    let mut labels = vec![-1i32; nc];

    let mut global_to_local = vec![NO_PARENT; big_brother.len()];
    for (li, &gi) in cc_idx.iter().enumerate() {
        global_to_local[gi] = li as i32;
    }

    let mut parent_local = vec![NO_PARENT; nc];
    for (li, &gi) in cc_idx.iter().enumerate() {
        let p = big_brother[gi];
        if p >= 0 {
            let p = p as usize;
            if p < global_to_local.len() {
                parent_local[li] = global_to_local[p];
            }
        }
    }
    for &c in centers {
        parent_local[c] = c as i32;
    }

    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); nc];
    for i in 0..nc {
        let p = parent_local[i];
        if p >= 0 {
            let pj = p as usize;
            if pj != i {
                adj[i].push(pj);
                adj[pj].push(i);
            }
        }
    }

    let mut comp_id = 0i32;
    let mut stack: Vec<usize> = Vec::new();

    for start in 0..nc {
        if labels[start] != -1 {
            continue;
        }
        labels[start] = comp_id;
        stack.push(start);

        while let Some(v) = stack.pop() {
            for &u in &adj[v] {
                if labels[u] == -1 {
                    labels[u] = comp_id;
                    stack.push(u);
                }
            }
        }
        comp_id += 1;
    }

    for v in labels.iter_mut() {
        *v += label_offset;
    }

    (comp_id, labels)
}
