import json
import os
import time
import random

from selenium import webdriver
from selenium.common import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

from Code import Alatius, CNN, LSTM


def bring_window_to_front(driver):
    """Bring the browser window to front and focus"""
    try:
        driver.switch_to.window(driver.current_window_handle)
        # Use JavaScript to focus the window
        driver.execute_script("window.focus();")
    except Exception as e:
        print(f"Could not bring window to front: {e}")

def check_if_game_over(driver):
    """Check if the round has ended"""
    # Check for "Try Again" button
    try_again_buttons = driver.find_elements(By.XPATH, "//button[contains(text(), 'Try Again')]")
    if len(try_again_buttons) > 0:
        return True

    # Check for leaderboard table
    leaderboard_tables = driver.find_elements(By.CLASS_NAME, "table-striped")  # Fill in the class name
    if len(leaderboard_tables) > 0:
        return True

    # If neither found, game is not over
    return False


def restart_game(driver):
    """Click the Try Again button to restart"""
    try:
        # Wait a moment for the screen to settle
        time.sleep(1)
        bring_window_to_front(driver)

        # Try multiple possible selectors for the Try Again button
        wait = WebDriverWait(driver, 2)

        # Method 1: Look for button with "Try Again" text
        try:
            try_again_btn = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Try Again')]"))
            )

            # Use JavaScript click to avoid interception
            driver.execute_script("arguments[0].click();", try_again_btn)
            print("Restarting game...")
            return True
        except TimeoutException:
            pass

        # Method 2: Look for button with specific class
        try:
            buttons = driver.find_elements(By.CSS_SELECTOR, "button.btn.btn-primary")
            for button in buttons:
                if "Try Again" in button.text or "again" in button.text.lower():
                    driver.execute_script("arguments[0].click();", button)
                    print("Restarting game...")
                    return True
        except:
            pass

        # Method 3: Just look for any prominent button on the results screen
        try:
            buttons = driver.find_elements(By.TAG_NAME, "button")
            for button in buttons:
                if button.is_displayed() and ("try" in button.text.lower() or "again" in button.text.lower()):
                    driver.execute_script("arguments[0].click();", button)
                    print("Restarting game...")
                    return True
        except:
            pass

        print("Could not find Try Again button")
        return False

    except Exception as e:
        print(f"Error restarting: {e}")
        return False


def click_random_button(driver):
    """Click a random answer button"""
    try:
        # Wait for buttons to be present
        time.sleep(0.5)

        buttons = driver.find_elements(By.CSS_SELECTOR, "button.btn.btn-dark.btn-block.btn-lg")

        # Filter to only visible, enabled buttons
        clickable_buttons = [btn for btn in buttons if btn.is_displayed() and btn.is_enabled()]

        if clickable_buttons:
            selected_button = random.choice(clickable_buttons)
            # Use JavaScript click to avoid interception
            driver.execute_script("arguments[0].click();", selected_button)
            return True

        print("No clickable buttons found")
        return False

    except Exception as e:
        print(f"Error clicking button: {e}")
        return False

def collect_latin_line(driver):
    try:
        wait = WebDriverWait(driver,10)
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

def find_and_click_pattern_button(driver, correct_pattern, click_wrong):
    """
    Find and click the button matching the pattern
    Returns 'correct', 'wrong', or 'unavailable'
    """
    try:
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

            # Terminate game if reached desired level
            if not click_wrong:
                for button, pattern in button_data:
                    if pattern == correct_pattern:
                        print(f"  ✓ Clicking CORRECT button: {pattern}")
                        driver.execute_script("arguments[0].click();", button)
                        return 'correct'

            # Click the correct button
            else:
                for button, pattern in button_data:
                    if pattern != correct_pattern:
                        print(f"  ✓ Clicking INCORRECT button: {pattern}")
                        driver.execute_script("arguments[0].click();", button)
                        return 'termination in progress'

        else:
            # Correct answer not available
            print(f"  ⚠ Correct answer NOT in options - clicking random")
            click_random_button(driver)
            return 'unavailable'

    except Exception as e:
        print(f"  Error: {e}")
        click_random_button(driver)
        return 'unavailable'

def load_existing_labeled_data(output_file):
    """
    Load already-processed data if it exists
    Returns dict mapping line -> full item
    """
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)

            return existing_data
        except:
            print(f"Could not load {output_file}, starting fresh")
            return {}, []
    else:
        print(f"No existing {output_file} found, starting fresh")
        return {}, []


if __name__ == "__main__":

    intended_target = int(input("Enter intended target: "))
    print("\nSelect translation type:")
    print("1. LSTM Model")
    print("2. CNN Model")
    print("3. Alatius Web Service")
    translator_type = int(input("Enter choice (1-3): "))

    # Set up the scansion function based on choice
    if translator_type == 1:
        print("Loading LSTM model...")
        model, char_to_id = LSTM.load_lstm_model()
        get_scansion = lambda line: LSTM.predict_lstm(model, char_to_id, line)
        print("LSTM model loaded!")

    elif translator_type == 2:
        print("Loading CNN model...")
        model, char_to_id = CNN.load_cnn_model()
        get_scansion = lambda line: CNN.predict_cnn(model, char_to_id, line)
        print("CNN model loaded!")

    else:
        print("Using Alatius web service...")
        get_scansion = Alatius.get_scansion_from_alatius

    driver = webdriver.Chrome()
    driver.get("https://hexameter.co/rapid-scan")
    print("Log in if needed. Press Enter to start...")
    input()

    output_file = "../Misc/hexameter_lines.json"
    time.sleep(1.0)

    data_collected = load_existing_labeled_data(output_file)

    correct_count = 0
    unavailable_count = 0
    total_count = 0
    last_line = None
    duplicate_count = 0

    i = 0
    click_wrong = False

    try:
        while True:
            time.sleep(1.2)

            if check_if_game_over(driver):
                i = 0
                click_wrong = False
                last_line = None
                duplicate_count = 0

                print(f"\n{'=' * 50}")
                print(f"ROUND COMPLETE")
                print(f"Correct answers: {correct_count}/{total_count}")
                print(f"Answer not available: {unavailable_count}/{total_count}")
                if total_count - unavailable_count > 0:
                    print(
                        f"Success rate when answer available: {correct_count / (total_count - unavailable_count) * 100:.1f}%")
                    print(f"Answer availability: {(total_count - unavailable_count) / total_count * 100:.1f}%")
                print(f"{'=' * 50}\n")

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data_collected, f, indent=2, ensure_ascii=False)

                if not restart_game(driver):
                    print("Failed to restart - exiting...")
                    break

                time.sleep(3.0)
                continue

            if intended_target <= i:
                click_wrong = True

            latin_line = collect_latin_line(driver)

            if not latin_line or len(latin_line) < 10:
                print("Failed to collect valid line")
                continue

            if latin_line == last_line:
                duplicate_count += 1
                print(f"Same line as previous (count: {duplicate_count})")

                if duplicate_count >= 3:
                    print("Stuck - clicking random and waiting longer...")
                    click_random_button(driver)
                    duplicate_count = 0
                    last_line = None
                continue
            else:
                duplicate_count = 0
                last_line = latin_line

            # Get correct pattern using selected method
            pattern = get_scansion(latin_line)

            if not pattern:
                print("Pattern failed, clicking random")
                click_random_button(driver)
            else:
                result = find_and_click_pattern_button(driver, pattern[:4], click_wrong)

            if click_wrong:
                continue

            time.sleep(.1)

            # Check if answer was correct
            blue_checks = driver.find_elements(By.CLASS_NAME, "fa-check-circle")
            if len(blue_checks) > 0:
                i = i + 1
                data_collected.append({
                    "line": latin_line,
                    "timestamp": time.time(),
                    "round": total_count + 1,
                    "pattern": pattern,
                    "pattern_first_4": pattern[:4]
                })

            total_count += 1
            if result == 'correct':
                correct_count += 1
                print(
                    f"📊 Score: {correct_count}/{total_count} | Available: {total_count - unavailable_count}/{total_count}")
            elif result == 'unavailable':
                unavailable_count += 1
                print(
                    f"📊 Score: {correct_count}/{total_count} | Available: {total_count - unavailable_count}/{total_count}")

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data_collected, f, indent=2, ensure_ascii=False)

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