/*
  Copyright© 2023 Raúl Wolters(1)

  This file is part of libinflx_rs (rust bindings for inflatox).

  inflatox is free software: you can redistribute it and/or modify it under
  the terms of the European Union Public License version 1.2 or later, as
  published by the European Commission.

  inflatox is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
  A PARTICULAR PURPOSE. See the European Union Public License for more details.

  You should have received a copy of the EUPL in an/all official language(s) of
  the European Union along with inflatox.  If not, see
  <https://ec.europa.eu/info/european-union-public-licence_en/>.

  (1) Resident of the Kingdom of the Netherlands; agreement between licensor and
  licensee subject to Dutch law as per article 15 of the EUPL.
*/

#[derive(Clone, Copy, Eq, Ord)]
#[repr(transparent)]
/// Data structure representing the inflatox version number. This version must
/// match between the python package and compiled artefact.
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

impl std::fmt::Debug for InflatoxVersion {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{self}")
  }
}

impl InflatoxVersion {
  pub const fn new(inner: [u16; 3]) -> Self {
    InflatoxVersion(inner)
  }
}
