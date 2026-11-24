use time::{Duration, OffsetDateTime, Time};
use time_tz::{OffsetDateTimeExt, timezones::db::america::NEW_YORK};

// These Eastern times bound the daily liquidation window.
pub const LIQUIDATION_START_TIME: Time = match Time::from_hms(15, 45, 0) {
    Ok(time) => time,
    Err(_) => panic!("The liquidation start time must be valid."),
};
pub const LIQUIDATION_END_TIME: Time = match Time::from_hms(20, 5, 0) {
    Ok(time) => time,
    Err(_) => panic!("The liquidation end time must be valid."),
};

// This preset retains the existing prices and order lifetimes used by the live bot.
pub const LIVE_CONFIG: Config = Config {
    discount_percent: 1.0_f64,
    markup_percent: 0.5_f64,
    buy_ttl: Duration::seconds(5),
    sell_ttl: Duration::seconds(120),
    expiry: Expiry::AtLifetime {
        liquidation_sell_ttl: Duration::minutes(1),
    },
    round_to_cents: true,
};

// These directions identify which resources a working order reserves.
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum Side {
    Buy,
    Sell,
}

// These values describe the prices and unreserved resources available for one decision.
pub struct Snapshot {
    pub bid_price: Option<f64>,
    pub ask_price: Option<f64>,
    pub buying_power: Option<f64>,
    pub sellable_shares: Option<f64>,
}

// This instruction requests a quantity at a specific price.
#[derive(Debug, PartialEq)]
pub struct Order {
    pub price: f64,
    pub shares: f64,
}

// A decision may place one order on each side independently.
#[derive(Debug, PartialEq)]
pub struct Placements {
    pub buy: Option<Order>,
    pub sell: Option<Order>,
}

// These presets pair each expiry boundary with its liquidation rules.
#[derive(Clone, Copy)]
enum Expiry {
    AtLifetime { liquidation_sell_ttl: Duration },
    AfterLifetime,
}

// These settings control pricing and expiration independently of order execution.
#[derive(Clone, Copy)]
pub struct Config {
    discount_percent: f64,
    markup_percent: f64,
    buy_ttl: Duration,
    sell_ttl: Duration,
    expiry: Expiry,
    round_to_cents: bool,
}

impl Config {
    // Supply the historical pricing and expiry conventions for one backtest candidate.
    pub fn backtest(
        discount_percent: f64,
        markup_percent: f64,
        buy_ttl: Duration,
        sell_ttl: Duration,
    ) -> Self {
        Self {
            discount_percent,
            markup_percent,
            buy_ttl,
            sell_ttl,
            expiry: Expiry::AfterLifetime,
            round_to_cents: false,
        }
    }

    // Price and size new orders using only the snapshot at the current decision.
    pub fn orders(&self, snapshot: &Snapshot, liquidating: bool) -> Placements {
        // Spend the available buying power on whole shares before the liquidation window.
        let buy = snapshot
            .bid_price
            .zip(snapshot.buying_power)
            .filter(|_| !liquidating)
            .and_then(|(bid, buying_power)| {
                let price = bid * (1.0_f64 - self.discount_percent / 100.0_f64);
                let price = if self.round_to_cents {
                    round_down_to_cent(price)
                } else {
                    price
                };
                let shares = (buying_power / price).floor();
                (price > 0.0_f64 && shares >= 1.0_f64).then_some(Order { price, shares })
            });

        // Offer all unreserved inventory, dropping the ask markup during liquidation.
        let offer = snapshot
            .ask_price
            .zip(snapshot.sellable_shares)
            .and_then(|(ask, shares)| {
                let price = if liquidating {
                    ask
                } else {
                    ask * (1.0_f64 + self.markup_percent / 100.0_f64)
                };
                let price = if self.round_to_cents {
                    (price * 100.0_f64).ceil() / 100.0_f64
                } else {
                    price
                };
                (price > 0.0_f64 && shares > 0.0_f64).then_some(Order { price, shares })
            });

        Placements { buy, sell: offer }
    }

    // Decide when to release or request cancellation of an already working order.
    pub fn should_cancel(&self, side: Side, age: Duration, liquidating: bool) -> bool {
        // Select the normal lifetime for this side before applying any liquidation rules.
        let lifetime = match side {
            Side::Buy => self.buy_ttl,
            Side::Sell => self.sell_ttl,
        };

        // Replay cancels every order during liquidation. Live lets sells rest briefly.
        match self.expiry {
            Expiry::AfterLifetime => liquidating || age > lifetime,
            Expiry::AtLifetime {
                liquidation_sell_ttl,
            } => {
                let lifetime = match (liquidating, side) {
                    (true, Side::Buy) => Duration::ZERO,
                    (true, Side::Sell) => liquidation_sell_ttl,
                    (false, _) => lifetime,
                };
                age >= lifetime
            }
        }
    }
}

// Use the supplied clock to identify the common Eastern liquidation window.
pub fn is_liquidating(now: OffsetDateTime) -> bool {
    let eastern_time = now.to_timezone(NEW_YORK).time();
    eastern_time >= LIQUIDATION_START_TIME && eastern_time < LIQUIDATION_END_TIME
}

// Floor dollar amounts to the cent boundary used for buying power and buy limits.
pub fn round_down_to_cent(price: f64) -> f64 {
    (price * 100.0_f64).floor() / 100.0_f64
}

#[cfg(test)]
mod tests {
    use super::{Config, LIVE_CONFIG, Side};
    use time::Duration;

    #[test]
    fn preserve_expiry_at_the_lifetime_boundary() {
        // Apply the existing live and replay boundaries to the same order lifetimes.
        let buy_ttl = Duration::seconds(5);
        let sell_ttl = Duration::seconds(120);
        let replay = Config::backtest(1.0_f64, 0.5_f64, buy_ttl, sell_ttl);
        for (side, ttl) in [(Side::Buy, buy_ttl), (Side::Sell, sell_ttl)] {
            assert!(!LIVE_CONFIG.should_cancel(side, ttl - Duration::NANOSECOND, false));
            assert!(LIVE_CONFIG.should_cancel(side, ttl, false));
            assert!(!replay.should_cancel(side, ttl, false));
            assert!(replay.should_cancel(side, ttl + Duration::SECOND, false));
        }
    }

    #[test]
    fn preserve_liquidation_cancellation_rules() {
        // Live cancels buys immediately and allows sells to rest for one minute.
        assert!(LIVE_CONFIG.should_cancel(Side::Buy, Duration::ZERO, true));
        assert!(!LIVE_CONFIG.should_cancel(Side::Sell, Duration::seconds(59), true));
        assert!(LIVE_CONFIG.should_cancel(Side::Sell, Duration::MINUTE, true));

        // Replay releases the full reservation for both sides on entering liquidation.
        let replay = Config::backtest(0.25_f64, 0.25_f64, Duration::HOUR, Duration::hours(4));
        assert!(replay.should_cancel(Side::Buy, Duration::ZERO, true));
        assert!(replay.should_cancel(Side::Sell, Duration::ZERO, true));
    }
}
