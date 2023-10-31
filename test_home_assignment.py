import unittest
import pandas as pd
from home_assignment import (add_day_diff, add_weekday, add_discount_diff, add_discount_perc)

class TestAssignment(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for data processing tests
        self.sample_data = {
            'Snapshot Date': ['1/1/2023', '1/2/2023'],
            'Checkin Date': ['1/5/2023', '1/7/2023'],
            'Original Price': [100, 200],
            'Discount Price': [80, 180]
        }
        self.df = pd.DataFrame(self.sample_data)

    def test_data_processing(self):
        # Test add_day_diff
        self.df = add_day_diff(self.df)
        expected_day_diff = [4, 5]
        self.assertTrue(self.df['DayDiff'].equals(pd.Series(expected_day_diff)))

        # Test add_weekday
        self.df = add_weekday(self.df)
        expected_weekdays = ['Thu', 'Sat']
        self.assertTrue(self.df['WeekDay'].equals(pd.Series(expected_weekdays)))

        # Test add_discount_diff
        self.df = add_discount_diff(self.df)
        expected_discount_diff = [20, 20]
        self.assertTrue(self.df['DiscountDiff'].equals(pd.Series(expected_discount_diff)))

        # Test add_discount_perc
        self.df = add_discount_perc(self.df)
        expected_discount_perc = [20.0, 10.0]
        self.assertTrue(self.df['DiscountPerc'].equals(pd.Series(expected_discount_perc)))

if __name__ == '__main__':
    unittest.main()
