import numpy as np
import pandas as pd
from greenvista.io import canopy_mask, points_in_cylinder, link_geoloc_volumes, load_volumes

def test_canopy_mask_excludes_ground_and_noise():
    cls = np.array([1, 2, 18, 1, 5])
    m = canopy_mask(cls)
    assert m.tolist() == [True, False, False, True, True]

def test_points_in_cylinder():
    xy = np.array([[0, 0], [0.5, 0], [3, 0]], float)
    z = np.array([1.0, 2.0, 3.0])
    sxy, sz = points_in_cylinder(xy, z, 0, 0, radius=1.0)
    assert len(sxy) == 2 and set(np.round(sz, 1)) == {1.0, 2.0}

def test_link_geoloc_volumes_keeps_only_matching_arv():
    # geoloc has 1,2,3,29 ; volumes (talhao 1) has 1,2,3 -> link keeps 3 (drops 29, the non-cubed point)
    geoloc = pd.DataFrame({"arv": [1, 2, 3, 29], "Easting": [0, 1, 2, 3], "Northing": [0, 0, 0, 0]})
    vol = pd.DataFrame({"arv": [1, 2, 3], "talhao": [1, 1, 1], "V": [0.1, 0.2, 0.3]})
    linked = link_geoloc_volumes(geoloc, vol, talhao=1)
    assert len(linked) == 3 and 29 not in linked["arv"].values

def test_load_volumes(tmp_path):
    p = tmp_path / "volumes.csv"
    pd.DataFrame({"arv": [1], "talhao": [1], "V": [0.5], "D_cm": [20], "H_m": [30]}).to_csv(p, index=False)
    df = load_volumes(p)
    assert df.loc[0, "V"] == 0.5
