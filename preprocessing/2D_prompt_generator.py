import os
import time
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Attempt of a automatic script to generate descriptions for 3D point cloud scenes using ChatGPT.
# The script was not used in practice, as there where issues with the account verification and the Selenium driver. 
# Instead i relied on the web-interface
# Easiest approach would be to use the original API, but as I had no ressources
# It is provided here for completeness.


service = Service(ChromeDriverManager().install())

# Set up undetectable Chrome options
chrome_options = uc.ChromeOptions()
chrome_options.add_argument("--start-maximized")  # Open in full-screen
chrome_options.add_argument("--disable-blink-features=AutomationControlled")  # Bypass bot detection
chrome_options.add_argument("--user-data-dir=C:/Users/YOUR_USER/AppData/Local/Google/Chrome/User Data")  # Use real Chrome profile
chrome_options.add_argument("--profile-directory=Default")  # Use default Chrome profile
chrome_options.add_argument("--no-sandbox")  # Bypass sandbox for stability
chrome_options.add_argument("--disable-dev-shm-usage")  # Prevent shared memory issues

# Launch undetectable ChromeDriver
driver = uc.Chrome(service=service, options=chrome_options)

# Open ChatGPT
driver.get("https://chat.openai.com/")

# Wait for manual login
input("Please log in to ChatGPT manually, then press Enter to continue...")

# Define dataset directory
dataset_dir = r"C:\Users\const\Projects\SAP_intro\scans"

# Define structured prompt
structured_prompt = """
You are assisting a **3D Point Cloud Understanding Model** by providing contextual guidance to improve its spatial reasoning. Your task is to identify and describe only the spatial relationships between key objects that a point cloud alone might struggle to infer.

**Follow this structured format for your response:**

Scene Type: <One or two words, e.g., 'Office', 'Living Room'>
Key Objects:
- <Object 1> (<Estimated Position>)
- <Object 2> (<Estimated Position>)
- <Object 3> (<Estimated Position>)
- <Object 4> (<Estimated Position>)
- <Object 5> (<Estimated Position>)

Spatial Arrangement Hint: <One sentence, only if object placement adds useful spatial context>
Object Interaction Clue: <One sentence, only if an object’s function or positioning is ambiguous>

**Context Guidelines:**
- This is **not** a replacement for 3D-based reasoning but a **supporting prompt** to enhance understanding.
- Focus only on **challenging spatial relationships** that may not be clear from the raw point cloud data.
- Avoid redundant details that a 3D model can already infer (e.g., basic object shapes, positions).
- Keep responses **concise and structured** for easy parsing.

Generate your response based on the given scene.
"""

# Loop through scene folders
for scene_id in os.listdir(dataset_dir):
    scene_folder = os.path.join(dataset_dir, scene_id)

    # Check if the image exists
    image_path = os.path.join(scene_folder, f"{scene_id}.color.png")
    if not os.path.exists(image_path):
        print(f"No image found for {scene_id}, skipping...")
        continue

    try:
        # Wait for the ChatGPT input box to become visible and interactable
        wait = WebDriverWait(driver, 20)  # Increased timeout for reliability
        input_box = wait.until(EC.visibility_of_element_located((By.TAG_NAME, "textarea")))
        input_box = wait.until(EC.element_to_be_clickable((By.TAG_NAME, "textarea")))

        # Ensure input box is interactable before sending keys
        time.sleep(2)  # Small delay to prevent misclicks
        try:
            input_box.click()
        except:
            driver.execute_script("arguments[0].scrollIntoView();", input_box)
            time.sleep(1)
            driver.execute_script("arguments[0].click();", input_box)

        # Send structured prompt
        input_box.send_keys(structured_prompt + Keys.RETURN)
        print(f"📨 Sent prompt for {scene_id}")

        # Wait dynamically for response
        response_wait_time = 20  # Maximum wait time for response
        start_time = time.time()

        while time.time() - start_time < response_wait_time:
            chat_messages = driver.find_elements(By.CLASS_NAME, "markdown")
            if chat_messages:
                break
            time.sleep(2)  # Check every 2 seconds

        # Scroll down to ensure full response is loaded
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)

        # Get the last ChatGPT response
        chat_messages = driver.find_elements(By.CLASS_NAME, "markdown")
        if chat_messages:
            description = chat_messages[-1].text.strip()
            print(f"✅ Received response for {scene_id}")
        else:
            print(f"⚠️ No response detected for {scene_id}, skipping...")
            continue

        # Save the description
        description_path = os.path.join(scene_folder, f"{scene_id}_description.txt")
        with open(description_path, "w", encoding="utf-8") as file:
            file.write(description)

        print(f"✅ Saved description for {scene_id}: {description_path}")

    except Exception as e:
        print(f"❌ Error processing {scene_id}: {str(e)}")

print("🎉 All descriptions retrieved and saved!")
