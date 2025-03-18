import zenoh
import time

# Store received messages
list_payload = []

def listener(sample):
    global list_payload
    message = sample.payload.to_string()
    print(f"Received {sample.kind} ('{sample.key_expr}': '{message}')")
    list_payload.append(message)  # Store messages

if __name__ == "__main__":
    with zenoh.open(zenoh.Config()) as session:
        sub = session.declare_subscriber('myhome/kitchen/temp', listener)
        
        try:
            while True:  # Keep running indefinitely
                time.sleep(1)
                print(f"list length: {len(list_payload)}")
                print(f"new value in the list: {list_payload[-1]}")
        except KeyboardInterrupt:
            print("\nSubscriber stopped.")