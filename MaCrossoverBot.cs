using cAlgo.API;
using cAlgo.API.Indicators;

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class MaCrossoverBot : Robot
    {
        [Parameter("Fast MA Source", Group = "Fast MA")]
        public DataSeries FastMaSource { get; set; }

        [Parameter("Fast MA Period", DefaultValue = 75, Group = "Fast MA", MinValue = 1)]
        public int FastMaPeriod { get; set; }

        [Parameter("Slow MA Source", Group = "Slow MA")]
        public DataSeries SlowMaSource { get; set; }

        [Parameter("Slow MA Period", DefaultValue = 200, Group = "Slow MA", MinValue = 1)]
        public int SlowMaPeriod { get; set; }

        [Parameter("Volume (Lots)", DefaultValue = 0.01, Group = "Trade", MinValue = 0.01)]
        public double VolumeInLots { get; set; }

        [Parameter("Stop Loss (Pips)", DefaultValue = 0, Group = "Trade", MinValue = 0)]
        public double StopLossInPips { get; set; }

        [Parameter("Take Profit (Pips)", DefaultValue = 0, Group = "Trade", MinValue = 0)]
        public double TakeProfitInPips { get; set; }

        [Parameter("Label", DefaultValue = "MaCrossoverBot", Group = "Trade")]
        public string Label { get; set; }

        private SimpleMovingAverage _fastMa;
        private SimpleMovingAverage _slowMa;
        private double _volumeInUnits;

        protected override void OnStart()
        {
            _volumeInUnits = Symbol.QuantityToVolumeInUnits(VolumeInLots);

            _fastMa = Indicators.SimpleMovingAverage(FastMaSource, FastMaPeriod);
            _slowMa = Indicators.SimpleMovingAverage(SlowMaSource, SlowMaPeriod);


        }

        protected override void OnBar()
        {
            // Ensure MA periods are valid for typical crossover logic, otherwise the signal meaning might be inverted
            // The HasCrossedAbove/Below methods will still function based on the actual values.
            // if (FastMaPeriod >= SlowMaPeriod)
            // {
            //     return; // Or handle as an inverted strategy if desired. For now, we allow it with a warning.
            // }

            bool isLongPositionOpen = Positions.Find(Label, SymbolName, TradeType.Buy) != null;

            // Buy Signal: Fast MA crosses above Slow MA
            if (_fastMa.Result.HasCrossedAbove(_slowMa.Result, 0))
            {
                if (!isLongPositionOpen)
                {
                    ExecuteMarketOrder(TradeType.Buy, SymbolName, _volumeInUnits, Label, StopLossInPips > 0 ? StopLossInPips : (double?)null, TakeProfitInPips > 0 ? TakeProfitInPips : (double?)null);
                }
            }
            // Sell Signal (Close Long): Fast MA crosses below Slow MA
            else if (_fastMa.Result.HasCrossedBelow(_slowMa.Result, 0))
            {
                if (isLongPositionOpen)
                {
                    var positionToClose = Positions.Find(Label, SymbolName, TradeType.Buy);
                    if (positionToClose != null) // Double check, though isLongPositionOpen should cover
                    {
                        ClosePosition(positionToClose);
                    }
                }
            }
        }
    }
}