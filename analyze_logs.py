import os
from collections import Counter

LOG_DIR = "logs"

def get_latest_log():
    files = [f for f in os.listdir(LOG_DIR) if f.endswith(".log")]
    if not files:
        print("❌ No log files found!")
        return None

    files.sort(reverse=True)  # latest first
    return os.path.join(LOG_DIR, files[0])


def analyze_log(log_file):
    levels = Counter()
    errors = []
    warnings = []
    cheating_events = Counter()

    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()

            # Count levels
            if "ERROR" in line:
                levels["ERROR"] += 1
                errors.append(line)

            elif "CRITICAL" in line:
                levels["CRITICAL"] += 1
                errors.append(line)

            elif "WARNING" in line:
                levels["WARNING"] += 1
                warnings.append(line)

                # Detect cheating type
                if "phone" in line.lower():
                    cheating_events["Phone Usage"] += 1
                elif "multiple" in line.lower():
                    cheating_events["Multiple Persons"] += 1
                elif "unknown" in line.lower():
                    cheating_events["Unknown Face"] += 1

            elif "INFO" in line:
                levels["INFO"] += 1

    return levels, errors, warnings, cheating_events


def print_report(levels, errors, warnings, cheating_events):
    print("\n" + "="*50)
    print("📊 PROCTORING LOG REPORT")
    print("="*50)

    print("\n📌 Log Summary:")
    for k, v in levels.items():
        print(f"{k}: {v}")

    print("\n🚨 Errors / Critical Issues:")
    if errors:
        for e in errors[:10]:
            print(" -", e)
    else:
        print("✔ No errors found")

    print("\n⚠️ Cheating Warnings:")
    if warnings:
        for w in warnings[:10]:
            print(" -", w)
    else:
        print("✔ No warnings")

    print("\n📊 Cheating Breakdown:")
    if cheating_events:
        for k, v in cheating_events.items():
            print(f"{k}: {v}")
    else:
        print("✔ No cheating detected")

    print("\n✅ System Status:")
    if levels["ERROR"] > 0 or levels["CRITICAL"] > 0:
        print("❌ System has issues (check errors)")
    else:
        print("✔ System running properly")


# ───────────────────────────────
# MAIN
# ───────────────────────────────
if __name__ == "__main__":
    log_file = get_latest_log()

    if log_file:
        print(f"\n📂 Analyzing: {log_file}")
        levels, errors, warnings, cheating_events = analyze_log(log_file)
        print_report(levels, errors, warnings, cheating_events)