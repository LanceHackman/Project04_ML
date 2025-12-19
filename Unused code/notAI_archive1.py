import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support.ui import Select

from collect_archive import check_if_game_over, restart_game, click_random_button


def collect_latin_line(driver):
    try:
        wait = WebDriverWait(driver, 10)
        time.sleep(2)
        line_element = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "default-latin-verse"))
        )
        latin_line = line_element.text.strip()
    except KeyboardInterrupt:
        print("Exiting")
        raise
    except Exception as e:
        print(f"Error: {e}")
        return None

    print("\nLine collected:", latin_line)
    return latin_line


def get_scansion_from_alatius(driver, latin_line):
    """
    Get scansion WITHOUT navigating to macronizer (assumes already there)
    Verifies the result is for the correct input line
    """
    try:
        wait = WebDriverWait(driver, 10)

        # Re-find the textarea
        textarea = wait.until(
            EC.presence_of_element_located((By.TAG_NAME, "textarea"))
        )

        # Clear and enter new text
        driver.execute_script("arguments[0].value = '';", textarea)
        time.sleep(0.2)
        textarea.send_keys(latin_line)

        # Re-find and select dropdown
        dropdown = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "select"))
        )
        select = Select(dropdown)
        select.select_by_visible_text("dactylic hexameters")

        # Click Submit
        submit_btn = driver.find_element(By.CSS_SELECTOR, "input[type='submit'][value='Submit']")
        submit_btn.click()

        # Wait for processing - look for the result section
        time.sleep(1.5)  # Give it time to process

        # Get the pattern
        feet_div = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "feet"))
        )
        pattern = feet_div.text.strip()

        # Verify the result is for OUR line by checking the macronized output
        # The macronized text appears after the pattern
        try:
            # Look for the macronized output (it should contain our line)
            result_section = driver.find_element(By.XPATH, "//h2[text()='Result']/following-sibling::p")
            result_text = result_section.text

            # Check if our input line is somewhere in the result (remove punctuation for comparison)
            input_words = latin_line.lower().replace(',', '').replace('.', '').split()[:3]
            result_lower = result_text.lower()

            # If first 3 words from input are in result, it's probably the right result
            if all(word in result_lower for word in input_words if len(word) > 2):
                print(f"  ✓ Verified result is for current line")
            else:
                print(f"  ⚠ Warning: Result might be stale")
                # Still try to use it but flag it
        except:
            print(f"  ⚠ Could not verify result freshness")

        # Check if first 4 characters are valid (D or S)
        if len(pattern) >= 4 and all(c in 'DS' for c in pattern[:4]):
            print(f"  New pattern: {pattern[:4]}")
            return pattern[:4]
        else:
            print(f"  Invalid pattern: {pattern}")
            return None

    except Exception as e:
        print(f"  Error getting scansion: {e}")
        return None


def find_and_click_pattern_button(driver, correct_pattern):
    """
    Find and click the button matching the pattern
    Returns 'correct', 'wrong', or 'unavailable'
    """
    try:
        time.sleep(0.3)

        wait = WebDriverWait(driver, 5)
        buttons = wait.until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "button.btn.btn-dark.btn-block.btn-lg"))
        )

        # Get button texts
        button_data = []
        for button in buttons:
            try:
                b_tag = button.find_element(By.TAG_NAME, "b")
                button_text = b_tag.text.strip()
                if button_text and len(button_text) >= 4:
                    button_data.append((button, button_text[:4]))
            except:
                continue

        available_patterns = [bd[1] for bd in button_data]
        print(f"  Correct pattern: {correct_pattern}")
        print(f"  Available buttons: {available_patterns}")

        # Check if correct answer is available
        if correct_pattern in available_patterns:
            # Click the correct button
            for button, pattern in button_data:
                if pattern == correct_pattern:
                    print(f"  ✓ Clicking CORRECT button: {pattern}")
                    driver.execute_script("arguments[0].click();", button)
                    return 'correct'
        else:
            # Correct answer not available
            print(f"  ⚠ Correct answer NOT in options - clicking random")
            click_random_button(driver)
            return 'unavailable'

    except Exception as e:
        print(f"  Error: {e}")
        click_random_button(driver)
        return 'unavailable'


if __name__ == "__main__":
    driver = webdriver.Chrome()

    # ----------- TAB 0: hexameter -----------
    driver.get("https://hexameter.co/rapid-scan")
    print("Log in if needed. Press Enter to start...")
    input()

    # ----------- TAB 1: alatius -----------
    driver.execute_script("window.open('about:blank');")

    alatius_tab = driver.window_handles[1]
    hexameter_tab = driver.window_handles[0]

    # Load macronizer once in tab 1
    driver.switch_to.window(alatius_tab)
    driver.get("https://alatius.com/macronizer/")
    time.sleep(2)

    # Switch back to game tab
    driver.switch_to.window(hexameter_tab)

    collected_lines = set()
    correct_count = 0
    unavailable_count = 0
    total_count = 0

    try:
        while True:
            driver.switch_to.window(hexameter_tab)

            if check_if_game_over(driver):
                print(f"\n{'=' * 50}")
                print(f"ROUND COMPLETE")
                print(f"Correct answers: {correct_count}/{total_count}")
                print(f"Answer not available: {unavailable_count}/{total_count}")
                if total_count - unavailable_count > 0:
                    print(
                        f"Success rate when answer available: {correct_count / (total_count - unavailable_count) * 100:.1f}%")
                    print(f"Answer availability: {(total_count - unavailable_count) / total_count * 100:.1f}%")
                print(f"{'=' * 50}\n")
                restart_game(driver)
                time.sleep(2)
                continue

            latin_line = collect_latin_line(driver)

            if not latin_line or len(latin_line) < 10:
                print("Failed to collect valid line")
                time.sleep(1)
                continue

            if latin_line in collected_lines:
                print(f"Duplicate line (skipping)")
                click_random_button(driver)
                total_count += 1
                continue

            collected_lines.add(latin_line)

            # Get correct pattern from macronizer
            driver.switch_to.window(alatius_tab)
            correct_pattern = get_scansion_from_alatius(driver, latin_line)
            driver.switch_to.window(hexameter_tab)

            if not correct_pattern:
                print("Pattern failed, clicking random")
                click_random_button(driver)
                total_count += 1
                time.sleep(0.5)
                continue

            # Try to find and click the correct button
            result = find_and_click_pattern_button(driver, correct_pattern)

            total_count += 1
            if result == 'correct':
                correct_count += 1
                print(
                    f"📊 Score: {correct_count}/{total_count} | Available: {total_count - unavailable_count}/{total_count}")
            elif result == 'unavailable':
                unavailable_count += 1
                print(
                    f"📊 Score: {correct_count}/{total_count} | Available: {total_count - unavailable_count}/{total_count}")

            time.sleep(0.8)

    except KeyboardInterrupt:
        print(f"\n\n{'=' * 50}")
        print("FINAL STATISTICS")
        print(f"Total questions: {total_count}")
        print(f"Correct answers: {correct_count}")
        print(f"Answer not available: {unavailable_count}")
        if total_count - unavailable_count > 0:
            print(f"Success rate (when available): {correct_count / (total_count - unavailable_count) * 100:.1f}%")
        if total_count > 0:
            print(f"Answer availability rate: {(total_count - unavailable_count) / total_count * 100:.1f}%")
        print(f"{'=' * 50}")

    finally:
        driver.quit()