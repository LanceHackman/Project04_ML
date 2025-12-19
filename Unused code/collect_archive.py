from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import json
import random



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

def collect_latin_lines_for_dataset():
    # Setup
    driver = webdriver.Chrome()
    driver.get("https://hexameter.co/rapid-scan")

    print("Please login manually if needed. Press Enter when ready to start...")
    input()

    data_collected = []
    rounds_completed = 0
    lines_this_round = 0

    try:
        while True:

            time.sleep(1.5)  # Brief pause

            # Check if game is over
            if check_if_game_over(driver):
                print(f"\n{'=' * 50}")
                print(f"Round {rounds_completed + 1} complete! Collected {lines_this_round} lines this round.")
                print(f"Total lines collected: {len(data_collected)}")
                print(f"{'=' * 50}\n")

                rounds_completed += 1
                lines_this_round = 0

                # Save data
                with open('../Misc/hexameter_lines.json', 'w', encoding='utf-8') as f:
                    json.dump(data_collected, f, indent=2, ensure_ascii=False)

                # Restart the game
                if not restart_game(driver):
                    print("Failed to restart, exiting...")
                    break

                continue

            # Wait for the Latin line to appear
            try:
                wait = WebDriverWait(driver, 10)
                line_element = wait.until(
                    EC.presence_of_element_located((By.CLASS_NAME, "default-latin-verse"))
                )

                # Extract the line
                latin_line = line_element.text.strip()

                # Check if we've already collected this line
                if not any(item['line'] == latin_line for item in data_collected):
                    print(f"New line #{len(data_collected) + 1}: {latin_line}")

                    # Store the data
                    data_collected.append({
                        "line": latin_line,
                        "timestamp": time.time(),
                        "round": rounds_completed + 1
                    })
                    lines_this_round += 1
                else:
                    print(f"Duplicate line (skipping): {latin_line}")

                # Click a random button to continue
                click_random_button(driver)

            except TimeoutException:
                print("Timeout waiting for line...")
                continue

            except Exception as e:
                print(f"Error: {e}")
                continue

            with open('../Misc/hexameter_lines.json', 'w', encoding='utf-8') as f:
                json.dump(data_collected, f, indent=2, ensure_ascii=False)

            print("Data saved to hexameter_lines.json")

    except KeyboardInterrupt:
        print(f"\n\n{'=' * 50}")
        print(f"Stopping collection.")
        print(f"Total lines collected: {len(data_collected)}")
        print(f"Rounds completed: {rounds_completed}")
        print(f"{'=' * 50}")

    finally:
        driver.quit()

if __name__ == "__main__":
    collect_latin_lines_for_dataset()