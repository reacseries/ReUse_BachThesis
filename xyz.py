from opcua import Client, ua

def setup_opcua_client(endpoint="opc.tcp://192.168.0.1:4840"):
    client = Client(endpoint)
    client.connect()
    print(f"[OPC UA] Connected to {endpoint}")

    # ✅ Parent array node (no index [0]/[2])
    node_tcp = client.get_node('ns=3;s="MotoLocal"."PosTCP"."TCPPosition"')

    # Optional test write
    try:
        test_values = [0.0, 0.0, 0.0]
        node_tcp.set_value(ua.Variant(test_values, ua.VariantType.Float))
        print("[OPC UA] Test write of full array succeeded.")
    except Exception as e:
        print("[OPC UA] ⚠️ Test write failed:", e)

    client.disconnect()

if __name__ == "__main__":
    setup_opcua_client()
