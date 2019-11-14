"""
Timezone definitions and conversion methods
"""
import datetime
import pytz

epoch = datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)


class FixedTimeZone(pytz._FixedOffset):
    """
    Class that represents a fixed time zone defined by UTC offset in hours.
    """
    def __init__(self, offset, name):
        """
        arg int offset: timezone UTC offset in hours
        arg str name: timezone name
        """
        self._offset_hours = offset
        offset_minutes = offset*60
        super(FixedTimeZone, self).__init__(offset_minutes)
        self.zone = name

    def tzname(self, dt):
        return self.zone

    def __repr__(self):
        return 'FixedTimeZone({:}, {:})'.format(self._offset_hours, self.zone)


def datetime_to_epoch(t):
    """
    Convert python datetime object to epoch time stamp.
    """
    return (t - epoch).total_seconds()


def epoch_to_datetime(t):
    """
    Convert python datetime object to epoch time stamp.
    """
    return epoch + datetime.timedelta(seconds=t)
