import os

def parse_dgd_line(line):
    parts = line.split()
    if len(parts) < 30:
        return None
    vals = [int(x) for x in parts]
    year, month, day = vals[0], vals[1], vals[2]
    A_middle, K_middle = vals[3], vals[4:12]
    A_high, K_high = vals[12], vals[13:21]
    A_planetary, K_planetary = vals[21], vals[22:30]
    return {
        'year': year,
        'month': month,
        'day': day,
        'A_middle': A_middle,
        'K_middle': K_middle,
        'A_high': A_high,
        'K_high': K_high,
        'A_planetary': A_planetary,
        'K_planetary': K_planetary
    }

def main():
    script_dir = os.path.dirname(__file__)
    fname = os.path.join(script_dir, "..", "data", "2015_DGD.txt")
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith(':'):
                continue
            parsed = parse_dgd_line(line)
            if parsed:
                print(
                    f"{parsed['year']:04d}-{parsed['month']:02d}-{parsed['day']:02d} "
                    f"Ap={parsed['A_planetary']}, "
                    f"Kp={parsed['K_planetary']}"
                )

if __name__ == "__main__":
    main()
