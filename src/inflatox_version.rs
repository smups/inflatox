#[derive(Clone, Copy, Eq, Ord)]
#[repr(transparent)]
pub(crate) struct InflatoxVersion([u16; 3]);

impl std::ops::Index<usize> for InflatoxVersion {
    type Output = u16;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl PartialOrd for InflatoxVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.0 != other.0 {
            self[0].partial_cmp(&other[0])
        } else if self[1] != other[1] {
            self[1].partial_cmp(&other[1])
        } else {
            self[2].partial_cmp(&other[2])
        }
    }
}

impl PartialEq for InflatoxVersion {
    fn eq(&self, other: &Self) -> bool {
        //Minor version does not matter
        (self[0] == other[0]) && (self[1] == other[1])
    }
}

impl std::fmt::Display for InflatoxVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}.{}.{}", self[0], self[1], self[2])
    }
}

impl InflatoxVersion {
    pub const fn new(inner: [u16; 3]) -> Self {
        InflatoxVersion(inner)
    }
}
