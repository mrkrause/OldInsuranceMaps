import math

from django.test import tag
from osgeo import gdal

from ohmg.georeference.georeferencer import Georeferencer
from ohmg.georeference.utils.gcps import (
    calculate_affine_rmse,
    calculate_helmert_rmse,
    get_helmert_params,
)

from .base import OHMGTestCase


@tag("warp")
class HelmertTransformationTestCase(OHMGTestCase):
    def test_helmert_transformation_calculations(self):
        """This test runs through 8 permutations of GCPs, and makes
        sure that the calculations used to set up the helmert
        transformations return the right values for every permutation.
        """

        # inner (smallest) angle for a 3,4,5 triangle
        theta_345 = math.degrees(math.asin(3 / 5))

        # variables that define positions and expected values to test
        data_matrix = [
            (0, 2.5, 0, -0.5, 3),
            (2, 1.5, 90 - theta_345, 2.1, 2.2),
            (2.5, 0, 90, 3, 0.5),
            (2, -1.5, 90 + theta_345, 2.7, -1.4),
            (0, -2.5, 180, 0.5, -3),
            (-2, -1.5, 270 - theta_345, -2.1, -2.2),
            (-2.5, 0, 270, -3, -0.5),
            (-2, 1.5, 270 + theta_345, -2.7, 1.4),
        ]

        # GCP 1 stays constant
        # GCP 2's image coords stay constant while the geo coords move
        # clockwise around the origin.
        gcp1 = gdal.GCP(0, 0, 0, 1, 6)
        for gcpx, gcpy, target_rotation, x_offset, y_offset in data_matrix:
            # note 1. GCP args are: geo x, geo y, geo z (not used), img x, img y
            # note 2: img y uses inverse y axis (per GCP convention)
            # note 3: The distance between img GCPs is 5 which creates a 3,4,5
            # triangle during rotated permutations of the test (helpful)
            # note 4: All geo GCP coords halve the dimensions of the triangle
            # so scale is .5
            gcp2 = gdal.GCP(gcpx, gcpy, 0, 1, 1)

            g = Georeferencer(crs="EPSG:3857", transformation="helmert", gcps_gdal=[gcp1, gcp2])

            params = get_helmert_params(g.gcps)
            self.assertAlmostEqual(params.scale, 0.5)
            self.assertAlmostEqual(params.rotation, target_rotation)
            self.assertAlmostEqual(params.offset_x, x_offset)
            self.assertAlmostEqual(params.offset_y, y_offset)


@tag("warp")
class RMSETestCase(OHMGTestCase):
    # Consider a square, with mapping errors of (+1, -1, -1, +1) * (0.3, 0.4)
    # at the corners. Neither the affine nor Helmert transform can absorb this
    # "twist" and all the error ends up in the residuals, which we can calculate
    # by hand. NB: Bias correction introduces a surprising result here: 
    # Helmert transform has lower (adjusted) RMSE than the affine transform!

    corners = [(0, 0), (100, 0), (0, 100), (100, 100)]
    signs = [1, -1, -1, 1]
    error = (0.3, 0.4)
    error_length = 0.5

    def true_coords(self, pixel, line):
        """Error-free geo coords: x = 1000 + 2 * line, y = 5000 + 2 * pixel."""
        return (1000 + 2 * line, 5000 + 2 * pixel)

    def make_gcps(self, error=(0, 0), corners=None):
        corners = corners or self.corners
        gcps = []
        for (pixel, line), sign in zip(corners, self.signs):
            x, y = self.true_coords(pixel, line)
            gcps.append(gdal.GCP(x + sign * error[0], y + sign * error[1], 0, pixel, line))
        return gcps

    def assert_coords_equal(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertAlmostEqual(a[0], e[0])
            self.assertAlmostEqual(a[1], e[1])

    def test_exact_fit(self):
        """An error-free transformation has zero RMSE."""
        gcps = self.make_gcps()
        self.assertAlmostEqual(calculate_affine_rmse(gcps)[0], 0, places=3)
        self.assertAlmostEqual(calculate_helmert_rmse(gcps)[0], 0, places=3)

    def test_known_residuals(self):
        """Per-point RMSE, corrected for degrees of freedom, with N = 4 and
        each point's error distance d = 0.5:
        affine:  sqrt(4 * d**2 / (4 - 3)) = 2 * d
        helmert: sqrt(4 * d**2 / (4 - 2)) = sqrt(2) * d
        """
        gcps = self.make_gcps(error=self.error)
        d = self.error_length
        self.assertAlmostEqual(calculate_affine_rmse(gcps)[0], 2 * d, places=3)
        self.assertAlmostEqual(calculate_helmert_rmse(gcps)[0], math.sqrt(2) * d, places=3)

    def test_predictions_and_lines(self):
        """The fits recover the error-free coords, and each line runs from a
        predicted coord to the GCP's (erroneous) geo coord."""
        gcps = self.make_gcps(error=self.error)
        true = [self.true_coords(pixel, line) for pixel, line in self.corners]
        observed = [(gcp.GCPX, gcp.GCPY) for gcp in gcps]
        for calculate in (calculate_affine_rmse, calculate_helmert_rmse):
            _, pred_coords, lines = calculate(gcps)
            self.assert_coords_equal(pred_coords, true)
            self.assert_coords_equal([line[0] for line in lines], true)
            self.assert_coords_equal([line[1] for line in lines], observed)

    def test_too_few_gcps(self):
        """
        When the fix is exactly determined, we return None, rather than a
        potentially misleading zero.
        """
        for calculate, n in ((calculate_affine_rmse, 3), (calculate_helmert_rmse, 2)):
            gcps = self.make_gcps(error=self.error, corners=self.corners[:n])
            rmse, pred_coords, lines = calculate(gcps)
            self.assertIsNone(rmse)
            self.assertEqual(len(pred_coords), n)
            self.assertEqual(len(lines), n)
