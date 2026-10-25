import unittest
from datetime import datetime, timezone

from penumbra.memory.read import time_window

NOW = datetime(2026, 10, 15, 8, 0, tzinfo=timezone.utc)


class TodayWindow(unittest.TestCase):
    def test_a_part_of_today_is_today_only(self):
        for q in ("上午什么时候喝过啦", "刚才说的那个", "下午去哪了", "今早吃了什么"):
            w = time_window(q, NOW)
            self.assertTrue(w and w.get("today"), q)
            self.assertEqual(w["after"], w["before"])

    def test_not_after_a_day_nor_beside_words_that_point_back(self):
        for q in ("那天下午", "周三早上", "上次早上你说", "还记得以前下午一起散步吗"):
            w = time_window(q, NOW)
            self.assertFalse(w and w.get("today"), q)

    def test_today_itself_keeps_its_old_window(self):
        w = time_window("今天好累", NOW)
        self.assertEqual(w["label"], "今天")
        self.assertNotIn("today", w)


if __name__ == "__main__":
    unittest.main()
