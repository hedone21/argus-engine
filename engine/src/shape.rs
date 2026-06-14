#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    pub fn numel(&self) -> usize {
        if self.dims.is_empty() {
            0
        } else {
            self.dims.iter().product()
        }
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self::new(dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_creation_and_metadata() {
        let shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(shape.ndim(), 3);
        assert_eq!(shape.dims(), &[2, 3, 4]);
        assert_eq!(shape.numel(), 24);

        let shape_from_vec: Shape = vec![2, 3, 4].into();
        assert_eq!(shape, shape_from_vec);
    }

    #[test]
    fn test_empty_shape_scalar() {
        let empty_shape = Shape::new(vec![]);
        assert_eq!(empty_shape.ndim(), 0);
        let empty_slice: &[usize] = &[];
        assert_eq!(empty_shape.dims(), empty_slice);
        assert_eq!(empty_shape.numel(), 0);
    }

    #[test]
    fn test_one_dimensional_empty() {
        let shape = Shape::new(vec![0]);
        assert_eq!(shape.ndim(), 1);
        assert_eq!(shape.numel(), 0);
    }
}
