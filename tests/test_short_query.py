import unittest
from unittest import mock

from penumbra import identity
from penumbra.memory.read import MIN_CONTENT, _content_length

# A deployment declares its own pet names ("filler_words" in profile.json); these are a stand-in.
PROFILE = identity.Identity(filler_words=("小熊", "亲亲宝"))


class ShortQueryTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(identity, "_current", PROFILE)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_pet_names_and_particles_do_not_count_as_content(self):
        self.assertLess(_content_length("小熊出来嘛 喂亲亲宝"), MIN_CONTENT)
        self.assertLess(_content_length("亲亲宝～"), MIN_CONTENT)

    def test_without_a_profile_only_particles_are_filler(self):
        with mock.patch.object(identity, "_current", identity.Identity()):
            self.assertGreaterEqual(_content_length("小熊出来嘛 喂亲亲宝"), MIN_CONTENT)

    def test_stage_directions_are_not_content(self):
        self.assertLess(_content_length("你猜呀（得意）"), 4)
        self.assertLess(_content_length("不开心（故意 撅着嘴）"), 4)
        self.assertGreaterEqual(_content_length("（跳过去）小熊人家之前告诉过你吃全熟牛排哒"), MIN_CONTENT)

    def test_a_message_with_a_topic_searches_on_its_own(self):
        self.assertGreaterEqual(_content_length("小熊明天去看海吗，想吃海鲜"), MIN_CONTENT)


if __name__ == "__main__":
    unittest.main()
