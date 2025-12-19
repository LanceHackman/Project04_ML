from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
import time
from selenium.webdriver.support.ui import Select
import os


def get_scansion_from_alatius(driver, latin_line):
    """
    Get scansion pattern from alatius.com/macronizer
    Returns pattern like "DDSSDS" or None if failed
    We only need the first 4 feet to be valid (D or S)
    """
    try:
        # Navigate to alatius macronizer
        driver.get("https://alatius.com/macronizer/")

        # Wait for page to load
        wait = WebDriverWait(driver, 10)

        # Find the textarea
        textarea = wait.until(
            EC.presence_of_element_located((By.TAG_NAME, "textarea"))
        )
        textarea.clear()
        textarea.send_keys(latin_line)

        # Select "dactylic hexameters" from dropdown
        dropdown = driver.find_element(By.CSS_SELECTOR, "select")
        select = Select(dropdown)
        select.select_by_visible_text("dactylic hexameters")

        # Click Submit
        submit_btn = driver.find_element(By.CSS_SELECTOR, "input[type='submit'][value='Submit']")
        submit_btn.click()

        time.sleep(0.5)

        # Wait for the feet div to appear
        feet_div = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "feet"))
        )

        pattern = feet_div.text.strip()

        # Check if first 4 characters are valid (D or S)
        if len(pattern) >= 4 and all(c in 'DS' for c in pattern[:4]):
            # Return full pattern but we know first 4 are good
            return pattern
        else:
            print(f"  Invalid pattern (first 4 chars): {pattern}")
            return None

    except Exception as e:
        print(f"  Error getting scansion: {e}")
        return None


def load_existing_labeled_data(output_file):
    """
    Load already-processed data if it exists
    Returns dict mapping line -> full item
    """
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)

            # Create a lookup dict by line text
            lookup = {item['line']: item for item in existing_data}
            print(f"Loaded {len(lookup)} already-labeled lines from {output_file}")
            return lookup, existing_data
        except:
            print(f"Could not load {output_file}, starting fresh")
            return {}, []
    else:
        print(f"No existing {output_file} found, starting fresh")
        return {}, []


def label_collected_data(input_file, output_file):
    """
    Process all collected lines and add scansion patterns
    Skips lines that are already in the output file
    Removes failed lines from the input file
    """
    # Load the collected lines
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} lines from {input_file}")

    # Load existing labeled data
    existing_lookup, labeled_data = load_existing_labeled_data(output_file)

    # Filter out lines that are already processed
    lines_to_process = [item for item in data if item['line'] not in existing_lookup]

    print(f"Already labeled: {len(existing_lookup)}")
    print(f"New lines to process: {len(lines_to_process)}")

    if len(lines_to_process) == 0:
        print("\nAll lines already processed!")
        return

    # Setup Selenium
    driver = webdriver.Chrome()

    failed_lines = []
    newly_labeled = 0

    try:
        for i, item in enumerate(lines_to_process):
            latin_line = item['line']
            print(f"\nProcessing {i + 1}/{len(lines_to_process)}: {latin_line}")

            # Get the scansion pattern
            pattern = get_scansion_from_alatius(driver, latin_line)

            if pattern:
                print(f"  Pattern: {pattern} (first 4: {pattern[:4]})")

                new_item = {
                    **item,  # Keep all original data
                    'pattern': pattern,
                    'pattern_first_4': pattern[:4]  # Store just the first 4 for easy access
                }

                labeled_data.append(new_item)
                existing_lookup[latin_line] = new_item  # Update lookup to avoid duplicates
                newly_labeled += 1
            else:
                print(f"  ✗ Failed to get pattern - will remove from input file")
                failed_lines.append(latin_line)

            # Save progress every 10 lines
            if (i + 1) % 10 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(labeled_data, f, indent=2, ensure_ascii=False)
                print(f"\n✓ Progress saved: {len(labeled_data)} total lines ({newly_labeled} new)")

            # Brief pause to be nice to the server
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        driver.quit()

        # Final save of labeled data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(labeled_data, f, indent=2, ensure_ascii=False)

        # Remove failed lines from input file
        if failed_lines:
            print(f"\nRemoving {len(failed_lines)} failed lines from {input_file}...")

            # Keep only lines that were successfully processed or not yet attempted
            failed_lines_set = set(failed_lines)
            cleaned_data = [item for item in data if item['line'] not in failed_lines_set]

            with open(input_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

            print(f"✓ Updated {input_file}: {len(data)} → {len(cleaned_data)} lines")

            # Save failed lines to a separate file for reference
            with open('hexameter_failed.json', 'a', encoding='utf-8') as f:
                for line in failed_lines:
                    f.write(json.dumps({"line": line, "timestamp": time.time()}) + '\n')
            print(f"✓ Failed lines saved to hexameter_failed.json")

        print(f"\n{'=' * 50}")
        print(f"Labeling complete!")
        print(f"Total labeled lines: {len(labeled_data)}")
        print(f"Newly labeled: {newly_labeled}")
        print(f"Failed (removed): {len(failed_lines)}")
        print(f"Data saved to: {output_file}")
        print(f"{'=' * 50}")

        if failed_lines:
            print("\nFailed lines (removed from input file):")
            for line in failed_lines[:10]:  # Show first 10
                print(f"  - {line}")
            if len(failed_lines) > 10:
                print(f"  ... and {len(failed_lines) - 10} more")


if __name__ == "__main__":
    label_collected_data('../hexameter_lines.json', 'hexameter_labeled.json')