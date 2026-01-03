#!/bin/bash
# Run validation checks every hour for 24 hours

VALIDATION_LOG="validation_24h_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$VALIDATION_LOG") 2>&1

echo "Starting hourly validation (24 checks over 24 hours)"
echo "Log: $VALIDATION_LOG"
echo ""

for hour in {1..24}; do
  echo ""
  echo "╔═══════════════════════════════════════════════════════════════╗"
  echo "║  HOURLY VALIDATION #$hour                                      "
  echo "╚═══════════════════════════════════════════════════════════════╝"
  echo ""
  echo "Time: $(date)"
  echo ""

  ./validation_checks.sh
  RESULT=$?

  if [ $RESULT -gt 10 ]; then
    echo ""
    echo "❌ CRITICAL FAILURE DETECTED"
    echo "   Failures: $RESULT"
    echo "   Action required: Investigation and potential restart"
    echo ""

    # Create alert file
    echo "CRITICAL_FAILURE at hour $hour" > ALERT_VALIDATION_FAILURE.txt
    echo "Failures: $RESULT" >> ALERT_VALIDATION_FAILURE.txt
    echo "Time: $(date)" >> ALERT_VALIDATION_FAILURE.txt

  elif [ $RESULT -gt 0 ]; then
    echo ""
    echo "⚠️  Minor issues detected"
    echo "   Failures: $RESULT"
    echo "   Continuing monitoring"
    echo ""
  else
    echo ""
    echo "✅ All checks passed"
    echo ""
  fi

  # Wait 1 hour (or exit if this is the last check)
  if [ $hour -lt 24 ]; then
    echo "Next validation in 1 hour..."
    sleep 3600
  fi
done

echo ""
echo "═══════════════════════════════════════"
echo "24-hour validation complete"
echo "Total validations: 24"
echo "Log: $VALIDATION_LOG"
echo "═══════════════════════════════════════"
