"""Time expressions in a memory question -> the date range it names (memory/read.py time_window)."""
import unittest
from datetime import datetime, timezone

from penumbra.memory.read import _residual, time_window

NOW = datetime(2026, 10, 2, 4, 0, tzinfo=timezone.utc)  # 2026-10-02 12:00 local


def span(q):
    w = time_window(q, NOW)
    return (w["after"], w["before"]) if w else None


class TimeWindowTest(unittest.TestCase):
    def test_parts_of_a_month(self):
        self.assertEqual(span("5月底那次"), ("2026-05-21", "2026-05-31"))
        self.assertEqual(span("九月中旬聊了什么"), ("2026-09-11", "2026-09-20"))
        self.assertEqual(span("8月初"), ("2026-08-01", "2026-08-10"))
        self.assertEqual(span("6月份我们去哪了"), ("2026-06-01", "2026-06-30"))
        self.assertEqual(span("2026年5月下旬"), ("2026-05-21", "2026-05-31"))

    def test_a_month_later_in_the_year_is_last_year(self):
        self.assertEqual(span("12月"), ("2025-12-01", "2025-12-31"))
        self.assertEqual(span("10月初"), ("2026-10-01", "2026-10-02"), "never past today")

    def test_relative_months(self):
        self.assertEqual(span("上个月初"), ("2026-09-01", "2026-09-10"))
        self.assertEqual(span("上个月"), ("2026-09-01", "2026-09-30"))
        self.assertEqual(span("月底的时候"), ("2026-09-21", "2026-09-30"), "early in a month, 月底 is the one that just ended")
        self.assertEqual(span("三个月前"), ("2026-07-01", "2026-07-31"))
        self.assertEqual(span("两个月前"), ("2026-08-01", "2026-08-31"))

    def test_existing_forms_still_work_and_no_false_month(self):
        self.assertEqual(span("9月28日去哪了"), ("2026-09-28", "2026-09-28"))
        self.assertEqual(span("昨天"), ("2026-10-01", "2026-10-01"))
        self.assertIsNone(span("月亮好圆"))
        self.assertIsNone(span("我们聊聊香菜"))

    def test_the_time_words_leave_the_topic(self):
        self.assertEqual(_residual("5月底的冰淇淋"), "冰淇淋")
        self.assertEqual(_residual("三个月前的香菜"), "香菜")


if __name__ == "__main__":
    unittest.main()
