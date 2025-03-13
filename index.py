import os
import json
import base64
import requests
import io
from PIL import Image

# OpenAI API Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

def encode_image(image_base64):
    """Process base64 image, resize to 512x512, and re-encode to base64."""
    try:
        img_data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to 512x512 maintaining aspect ratio
        img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        
        # Create new image with padding
        new_img = Image.new('RGB', (512, 512), (255, 255, 255))  # White background
        offset = ((512 - img.size[0]) // 2, (512 - img.size[1]) // 2)
        new_img.paste(img, offset)
        
        # Convert back to base64
        buffer = io.BytesIO()
        new_img.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

def main(context):
    """
    Analyze clothing items using GPT-4 Vision API.
    
    Args:
        context: The Appwrite function context containing request data.
    
    Returns:
        JSON response with clothing analysis results.
    """
    print("\nStarting GPT-4 Vision processing.\n")
    
    try:
        print("Extracting request context...")
        
        # Debug context structure
        print(f"Context type: {type(context)}")
        print(f"Context dir: {dir(context)}")
        print(f"Request object: {context.req}")
        print(f"Request dir: {dir(context.req)}")
        
        # Get request body
        body_str = context.req.body
        print(f"Raw body: {body_str}")
        
        try:
            body_data = json.loads(body_str)
            images = body_data.get('images', [])
            print(f"Parsed JSON: {body_data}")
        except (json.JSONDecodeError, TypeError):
            print("Failed to parse body as JSON")
            return context.res.json({
                "success": False,
                "message": "Invalid JSON format in request body"
            })
        
        if not images:
            return context.res.json({
                "success": False,
                "message": "No images provided"
            })
        
        if not isinstance(images, list):
            images = [images]  # Ensure it's always a list
        
        print(f"Received {len(images)} images for processing.")
        
        # Ensure API key exists
        if not OPENAI_API_KEY:
            print("Missing OpenAI API Key")
            return context.res.json({
                "success": False,
                "message": "OPENAI_API_KEY is missing from environment variables"
            })

        # Process images
        processed_images = [encode_image(img) for img in images]

        system_prompt = """You are a fashion expert analyzing clothing items in images.
        For each item, identify:
        1. Type of clothing/accessory
        2. Brand (if visible or recognizable, otherwise suggest a similar brand)
        3. Color (be specific with shades)
        4. Material (if visible, otherwise suggest the most likely material)
        5. Aesthetic/style (e.g., casual, formal, streetwear)
        6. Extra details (specific item or close alternative)
        7. Item Name (The name of the clothing item. Format in Title Case.)
        
        Format response as a JSON array with:
        {
            "type": "",
            "brand": "",
            "color": "",
            "material": "",
            "aesthetic": "",
            "extra_details": "",
            "item_name": "",
            "confidence": 0.0
        }

        Return results in the same order as provided images. Never use 'unknown' - suggest alternatives instead.
        Pay attention to logos to help you find the correct brand.

        You are NOT allowed to return any thing as 'Unknown'. Take your best shot and give your best guess. NEVER MARK SOMETHING AS UNKNOWN.
        """

        # Construct API request payload
        content = [{"type": "text", "text": f"Analyze {len(processed_images)} clothing items and return structured JSON."}]
        for base64_image in processed_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}],
            "max_tokens": 2000
        }

        print(f"Sending request to OpenAI API with payload: {json.dumps(payload)[:500]}...")

        response = requests.post(OPENAI_API_URL, headers=headers, json=payload)
        print(f"Response status code: {response.status_code}")
        
        if response.status_code != 200:
            error_message = f"API request failed with status code {response.status_code}: {response.text}"
            print(error_message)
            return context.res.json({
                "success": False,
                "message": error_message
            })
        
        print("Successfully received response from OpenAI API.")
        
        # Extract response data
        response_json = response.json()
        response_content = response_json['choices'][0]['message']['content']

        # Cleanup and parse JSON
        if "```json" in response_content:
            response_content = response_content.split("```json")[1].split("```")[0].strip()
        elif "```" in response_content:
            response_content = response_content.split("```")[1].split("```")[0].strip()
        
        analyses = json.loads(response_content)
        if not isinstance(analyses, list):
            analyses = [analyses]

        # Add confidence scores
        for analysis in analyses:
            analysis["confidence"] = 0.9

        print("Returning final JSON response.")

        return context.res.json({
            "success": True,
            "analyses": analyses
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return context.res.json({
            "success": False,
            "message": f"Unexpected error: {str(e)}"
        })
