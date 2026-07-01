from unittest.mock import patch

from django.test import Client, TestCase

from main.camera_utils import open_camera


class CameraUtilsTests(TestCase):
    def test_open_camera_uses_requested_index_when_available(self):
        with patch("main.camera_utils.cv2.VideoCapture") as mock_capture:
            mock_capture.return_value.isOpened.return_value = True

            cap = open_camera(camera_indices=[2, 3])

            self.assertIs(cap, mock_capture.return_value)
            self.assertEqual(mock_capture.call_args_list[0].args[0], 2)

    def test_open_camera_falls_back_to_next_index(self):
        with patch("main.camera_utils.cv2.VideoCapture") as mock_capture:
            class Cap:
                def __init__(self, opened):
                    self._opened = opened

                def isOpened(self):
                    return self._opened

                def set(self, prop, value):
                    return True

            first_cap = Cap(False)
            second_cap = Cap(True)
            mock_capture.side_effect = [first_cap, second_cap]

            cap = open_camera(camera_indices=[1, 2])

            self.assertIs(cap, second_cap)
            self.assertEqual(mock_capture.call_count, 2)
            self.assertEqual(mock_capture.call_args_list[1].args[0], 2)


class DashboardViewTests(TestCase):
    def test_dashboard_includes_current_session_time(self):
        client = Client()

        response = client.get("/dashboard/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("now_time", response.context)
        self.assertIn("today", response.context)
